"""
Implementation of the ARROU methodology proposed in the mid‑project report
for the advanced NLP course.

This module ties together three key phases:

1. **Layer discovery via LAT** – search for model layers that store the
   target knowledge to be unlearned. We expose a `discover_layers` function
   that uses a projected gradient descent (PGD) adversary to compute attack
   success rates (ASR) for a set of candidate layers. Layers with the
   highest ASR values are selected as critical storage locations.

2. **Stable unlearning via r‑ROME** – apply a rank–one update at the
   discovered layer to redirect the model away from the original answer and
   toward a refusal response. We implement helpers to compute the key and
   value vectors and to perform the r‑ROME update in a numerically stable
   way, following Eq. 9–10 in the report【185935615541641†L390-L406】.

3. **Adversarial robustness enhancement** – further harden the edit by
   incorporating adversarial contexts into the key estimation and by
   projecting the value away from adversarial subspaces【185935615541641†L459-L466】. The
   `robust_unlearning` function demonstrates how to recompute the key
   vector with both clean and adversarial activations and optionally project
   the value vector away from attack directions.

This code is designed to be illustrative rather than production‑ready. It
assumes access to a HuggingFace‐compatible transformer model and tokenizer
on the caller side. Heavy computations (e.g. PGD attacks) are separated
into helper functions so that they can be replaced by more efficient
implementations (for instance, from the `lat-train` repository) without
modifying the high‑level logic here.

Example usage:

>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> from arro_methodology import discover_layers, compute_key_vector, compute_value_vector, r_rome_update
>>> model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3-8B')
>>> tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3-8B')
>>> # assume dataloader yields dicts with input_ids and labels
>>> critical_layers = discover_layers(model, dataloader, candidate_layers=[4,8,12], epsilon=2.0)
>>> ke = compute_key_vector(model, tokenizer, subject_prompt="Who wrote The Hobbit?", layer=critical_layers[0])
>>> v_star = compute_value_vector(model, tokenizer, refusal_text="I don't know", layer=critical_layers[0])
>>> r_rome_update(model, layer=critical_layers[0], key_vector=ke, value_vector=v_star)

The edited model will now refuse questions about the subject specified in
`subject_prompt` while preserving its performance on unrelated tasks.
"""

from __future__ import annotations

import math
import torch
from torch import nn
from typing import Iterable, List, Tuple, Callable, Dict, Any

####################################################################################################
# Phase 1: Layer discovery via LAT
####################################################################################################

def _pgd_attack(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    layer: int,
    epsilon: float,
    learning_rate: float,
    iterations: int,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device | str = "cuda",
) -> Tuple[float, torch.Tensor]:
    """
    Run a simple PGD attack on a specified layer to maximize the model's
    likelihood of the original answer. This helper is deliberately
    lightweight: it adds a learnable perturbation to the hidden states
    returned by the specified layer and performs projected gradient ascent
    to increase the target loss.

    Args:
        model: The transformer model.
        batch: Dictionary containing `input_ids` and `labels` tensors.
        layer: Index of the transformer layer to attack.
        epsilon: L∞ radius for the perturbation.
        learning_rate: Step size for the PGD update.
        iterations: Number of PGD steps to perform.
        loss_fn: Function taking (logits, labels) → scalar loss; should
            encourage generation of the original (pre‑unlearning) answer.
        device: Device to run the attack on.

    Returns:
        A tuple `(asr, perturbation)` where `asr` is the attack success rate
        (fraction of examples where the attack flips the answer) and
        `perturbation` is the learned adversarial perturbation tensor.
    """
    model.eval()
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    
    # Get model's dtype to ensure consistency
    model_dtype = next(model.parameters()).dtype
    
    # Forward hook to inject perturbation
    perturb = torch.zeros(1, 1, model.config.hidden_size, device=device, dtype=model_dtype, requires_grad=True)
    asr = 0.0

    def hook_fn(module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
        # Broadcast perturbation across batch and sequence
        return output + perturb

    # Register hook on the MLP submodule of the specified layer
    handle = model.model.layers[layer].mlp.register_forward_hook(hook_fn)
    optim = torch.optim.Adam([perturb], lr=learning_rate)

    for _ in range(iterations):
        optim.zero_grad()
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
        loss = loss_fn(logits, labels)
        # Gradient ascent on loss
        (-loss).backward()
        optim.step()
        # Project perturbation back to L∞ ball
        with torch.no_grad():
            perturb.data = torch.clamp(perturb.data, -epsilon, epsilon)

    # Evaluate attack success: whether perturbation changes model's most likely answer
    with torch.no_grad():
        # Baseline (no perturbation)
        handle.remove()
        base_outputs = model(input_ids=input_ids)
        base_preds = base_outputs.logits.argmax(dim=-1)
        # Apply perturbation one last time for evaluation
        handle = model.model.layers[layer].mlp.register_forward_hook(hook_fn)
        adv_outputs = model(input_ids=input_ids)
        adv_preds = adv_outputs.logits.argmax(dim=-1)
        asr = (adv_preds != base_preds).float().mean().item()
        handle.remove()

    return asr, perturb.detach()


def discover_layers(
    model: nn.Module,
    dataloader: Iterable[Dict[str, torch.Tensor]],
    candidate_layers: Iterable[int],
    epsilon: float = 2.0,
    pgd_lr: float = 5e-2,
    pgd_iters: int = 16,
    device: torch.device | str = "cuda",
) -> List[int]:
    """
    Identify which transformer layers store the knowledge to be unlearned.
    This function iterates over a set of candidate layers, performs a PGD
    attack on each layer using a few batches from the dataloader, and
    computes the attack success rate (ASR) defined in Eq. 6
    of the mid‑project report【185935615541641†L352-L359】. Layers with the highest ASR values
    are returned.

    Args:
        model: Transformer model to analyse.
        dataloader: Iterable yielding dictionaries with `input_ids` and
            `labels`. Only a few batches are used to estimate ASR.
        candidate_layers: Sequence of layer indices to probe.
        epsilon: Maximum L∞ norm of the adversarial perturbation.
        pgd_lr: Learning rate for the PGD attack.
        pgd_iters: Number of PGD steps per batch.
        device: Device for computation.

    Returns:
        A list of layers ranked by attack success rate (descending).
    """
    layer_asr: Dict[int, float] = {l: 0.0 for l in candidate_layers}
    # We'll use a simple cross‑entropy loss to encourage the original answer
    ce_loss = nn.CrossEntropyLoss()
    for batch_idx, batch in enumerate(dataloader):
        # Limit the number of batches used for efficiency
        if batch_idx >= 2:
            break
        for layer in candidate_layers:
            def loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                # Flatten for CE loss; shift labels to align with decoder predictions
                shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
                shift_labels = labels[..., 1:].contiguous().view(-1)
                return ce_loss(shift_logits, shift_labels)
            asr, _ = _pgd_attack(
                model=model,
                batch=batch,
                layer=layer,
                epsilon=epsilon,
                learning_rate=pgd_lr,
                iterations=pgd_iters,
                loss_fn=loss_fn,
                device=device,
            )
            # Accumulate ASR across batches
            layer_asr[layer] += asr
    # Normalize ASR by number of batches considered
    num_batches = min(2, batch_idx + 1)
    for layer in layer_asr:
        layer_asr[layer] /= num_batches
    # Sort layers by descending ASR
    ranked_layers = sorted(layer_asr, key=lambda l: layer_asr[l], reverse=True)
    return ranked_layers

####################################################################################################
# Phase 2: Stable unlearning via r‑ROME
####################################################################################################

@torch.no_grad()
def compute_key_vector(
    model: nn.Module,
    tokenizer: Any,
    subject_prompt: str,
    layer: int,
    num_prefixes: int = 10,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """
    Compute the averaged key representation for the given subject at a
    particular layer, following Eq. 7 of the report【185935615541641†L372-L381】. We sample
    random prefix tokens and append the subject prompt, then take the
    hidden activations at the specified layer and average them.

    Args:
        model: Transformer model.
        tokenizer: Tokenizer corresponding to the model.
        subject_prompt: The prompt containing the subject entity (e.g.,
            ``"Who wrote The Hobbit?"``). The last token of the prompt is
            assumed to be the subject token.
        layer: Layer index at which to collect activations.
        num_prefixes: Number of random prefixes to sample.
        device: Device for computation.

    Returns:
        A tensor of shape (hidden_size,) representing the averaged key vector.
    """
    model.eval()
    # Tokenize the subject prompt once
    subject_ids = tokenizer(subject_prompt, return_tensors="pt").input_ids.to(device)
    hidden_size = model.config.hidden_size
    
    # Get model's dtype to ensure consistency
    model_dtype = next(model.parameters()).dtype
    
    key_accum = torch.zeros(hidden_size, device=device, dtype=model_dtype)
    # Randomly sample prefixes from the tokenizer's vocab
    vocab = list(range(tokenizer.vocab_size))
    for i in range(num_prefixes):
        # Sample a random prefix length between 1 and 5 tokens
        prefix_length = torch.randint(low=1, high=5, size=(1,)).item()
        # Sample a prefix on the correct device.  Using torch.randint with a
        # 2D shape avoids inadvertently constructing a 1D tensor that would
        # cause dimension mismatches when concatenated below.  See issue
        # encountered during unlearning where ``torch.cat`` failed because
        # ``prefix_ids`` was 1D.
        prefix_ids = torch.randint(
            low=0,
            high=len(vocab),
            size=(1, prefix_length),
            device=device,
        )
        # Concatenate prefix and subject prompt along sequence dimension
        input_ids = torch.cat([prefix_ids, subject_ids], dim=1)
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        # Take the hidden state at the final position of the subject token
        h = outputs.hidden_states[layer][0, -1]  # shape: (hidden_size,)
        key_accum += h
        
        # Memory optimization: clear intermediate tensors every few iterations
        if (i + 1) % 3 == 0:
            del outputs, h
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return key_accum / num_prefixes


@torch.no_grad()
def compute_value_vector(
    model: nn.Module,
    tokenizer: Any,
    refusal_text: str,
    layer: int,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """
    Compute the value vector (target activation) for the refusal response,
    corresponding to Eq. 8 in the report【185935615541641†L385-L388】. We feed the refusal
    text through the model and take the hidden state of the last token at
    the specified layer.

    Args:
        model: Transformer model.
        tokenizer: Tokenizer.
        refusal_text: Text to use as the target (e.g., ``"I don't know"``).
        layer: Layer index from which to extract the activation.
        device: Device for computation.

    Returns:
        A tensor of shape (hidden_size,) representing the value vector.
    """
    model.eval()
    ids = tokenizer(refusal_text, return_tensors="pt").input_ids.to(device)
    outputs = model(input_ids=ids, output_hidden_states=True)
    v_star = outputs.hidden_states[layer][0, -1].clone()
    return v_star


def r_rome_update(
    model: nn.Module,
    layer: int,
    key_vector: torch.Tensor,
    value_vector: torch.Tensor,
    cov_matrix: torch.Tensor | None = None,
    device: torch.device | str = "cuda",
) -> None:
    """
    Perform the r‑ROME rank–one update on the weight matrix of the MLP
    feed‑forward block at the specified layer. The update formula is given
    by Eq. 9–10 in the report【185935615541641†L389-L401】. If a covariance matrix
    ``cov_matrix`` is provided, it is used as \(C_0\); otherwise the identity
    matrix is assumed.

    Args:
        model: Transformer model to edit. The update is performed in place.
        layer: Layer index whose MLP weight matrix is updated.
        key_vector: Averaged key vector ``k_e`` of shape (hidden_size,).
        value_vector: Target value vector ``v*`` of shape (hidden_size,).
        cov_matrix: Optional covariance matrix of shape (hidden_size,
            hidden_size). Defaults to identity if None.
        device: Device for computation.
    """
    # Get current weight matrix W (d_out × d_in) from the MLP
    target_layer = model.model.layers[layer].mlp
    weight = target_layer.down_proj.weight  # shape: (hidden_size, intermediate_size)
    
    # Get model's dtype
    model_dtype = weight.dtype
    
    # Pass key_vector through the first part of MLP to get intermediate representation
    with torch.no_grad():
        k_input = key_vector.unsqueeze(0).to(device=device, dtype=model_dtype)  # shape: (1, hidden_size)
        gate_output = target_layer.act_fn(target_layer.gate_proj(k_input))
        up_output = target_layer.up_proj(k_input)
        k_intermediate = (gate_output * up_output).squeeze(0)  # shape: (intermediate_size,)
    
    # Ensure key and value are column vectors in the intermediate space
    k = k_intermediate.view(-1, 1).to(device=device, dtype=model_dtype)  # shape: (intermediate_size, 1)
    v = value_vector.view(-1, 1).to(device=device, dtype=model_dtype)  # shape: (hidden_size, 1)
    # Default covariance matrix (in intermediate space)
    if cov_matrix is None:
        cov_matrix = torch.eye(k.size(0), device=device, dtype=model_dtype)
    # Compute numerator and denominator according to Eq. 9
    # Δ = (v* − W k_e) (k_e^T C_0^{-1}) / (k_e^T C_0^{-1} k_e)
    # Note: torch.inverse doesn't support FP16, so we need to cast to FP32
    Wk = weight @ k  # shape: (hidden_size, 1)
    
    # Cast to FP32 for inversion operations
    k_fp32 = k.float()
    v_fp32 = v.float()
    Wk_fp32 = Wk.float()
    cov_inv_fp32 = torch.inverse(cov_matrix.float())
    
    numerator = (v_fp32 - Wk_fp32) @ (k_fp32.t() @ cov_inv_fp32)
    denominator = (k_fp32.t() @ cov_inv_fp32 @ k_fp32).item()
    delta = (numerator / denominator).to(dtype=model_dtype)
    
    # Update weight
    with torch.no_grad():
        weight.add_(delta)


####################################################################################################
# Phase 3: Adversarial robustness enhancement
####################################################################################################

def robust_unlearning(
    model: nn.Module,
    tokenizer: Any,
    subject_prompt: str,
    refusal_text: str,
    layer: int,
    adversarial_contexts: List[Tuple[str, torch.Tensor]] | None = None,
    alpha: float = 0.1,
    num_prefixes: int = 10,
    device: torch.device | str = "cuda",
) -> None:
    """
    Apply r‑ROME unlearning and optionally incorporate adversarial contexts
    to harden the edit. This function follows the procedure outlined in
    Sec. 4.2.3 of the mid‑project report【185935615541641†L446-L465】.

    Steps:
      1. Compute the clean key vector for the subject (Eq. 7).
      2. If adversarial contexts are provided, average their hidden
         activations with the clean key to obtain a robust key (Eq. 12).
      3. Compute the value vector for the refusal text (Eq. 8).
      4. Optionally project the value away from the adversarial attack
         subspace (Eq. 13) using a simple orthogonal projection.
      5. Apply the r‑ROME update in place.

    Args:
        model: Transformer model to edit.
        tokenizer: Tokenizer.
        subject_prompt: Question containing the subject to unlearn.
        refusal_text: Text that the model should output instead.
        layer: Layer index to edit.
        adversarial_contexts: Optional list of tuples
            `(context_str, delta)` where `context_str` is an adversarially
            crafted prompt and `delta` is the learned latent perturbation.
        alpha: Scaling factor for projecting value away from the attack
            directions. A higher alpha pushes the value further from the
            adversarial subspace.
        num_prefixes: Number of random prefixes to use in key vector computation.
        device: Device for computation.
    """
    # Compute clean key
    k_clean = compute_key_vector(
        model=model,
        tokenizer=tokenizer,
        subject_prompt=subject_prompt,
        layer=layer,
        num_prefixes=num_prefixes,
        device=device,
    )
    # Incorporate adversarial contexts into the key (Eq. 12)
    if adversarial_contexts:
        total = k_clean.clone()
        model_dtype = next(model.parameters()).dtype
        for ctx_str, delta in adversarial_contexts:
            # Feed adversarial context plus perturbation through the model
            ids = tokenizer(ctx_str, return_tensors="pt").input_ids.to(device)
            # Ensure delta has the correct dtype
            delta = delta.to(dtype=model_dtype, device=device)
            def hook_fn(module: nn.Module, inp, out):
                return out + delta
            handle = model.model.layers[layer].mlp.register_forward_hook(hook_fn)
            with torch.no_grad():
                outputs = model(input_ids=ids, output_hidden_states=True)
                h_adv = outputs.hidden_states[layer][0, -1]
            handle.remove()
            total += h_adv
        k_robust = total / (1 + len(adversarial_contexts))
    else:
        k_robust = k_clean
    # Compute value vector
    v_star = compute_value_vector(
        model=model,
        tokenizer=tokenizer,
        refusal_text=refusal_text,
        layer=layer,
        device=device,
    )
    # Project value away from attack subspace (Eq. 13)
    if adversarial_contexts:
        # Build attack direction matrix V_attack (columns are normalized deltas)
        model_dtype = next(model.parameters()).dtype
        attack_dirs = []
        for _, delta in adversarial_contexts:
            # Ensure delta has correct dtype
            delta = delta.to(dtype=model_dtype, device=device)
            d = delta.view(-1)
            if d.norm() > 0:
                attack_dirs.append(d / d.norm())
        if attack_dirs:
            V = torch.stack(attack_dirs, dim=1).to(device)  # shape: (hidden_size, K)
            # Orthogonal projector onto attack subspace
            # Note: torch.inverse doesn't support FP16, so we need to cast to FP32
            V_fp32 = V.float()
            v_star_fp32 = v_star.float()
            proj_mat = V_fp32 @ torch.inverse(V_fp32.t() @ V_fp32) @ V_fp32.t()
            # Project v_star onto attack subspace and subtract
            v_proj = proj_mat @ v_star_fp32
            v_star = (v_star_fp32 - alpha * v_proj).to(dtype=model_dtype)
    # Apply r‑ROME update with robust key and modified value
    r_rome_update(
        model=model,
        layer=layer,
        key_vector=k_robust,
        value_vector=v_star,
        cov_matrix=None,
        device=device,
    )


__all__ = [
    "discover_layers",
    "compute_key_vector",
    "compute_value_vector",
    "r_rome_update",
    "robust_unlearning",
]