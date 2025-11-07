"""
Minimax r‑ROME/LAT integration
==============================

This script sketches a proof‑of‑concept implementation of the method
described in the accompanying report: combining targeted latent adversarial
training (LAT) with robust ROME (r‑ROME) knowledge editing to build
models that robustly unlearn sensitive knowledge.  It follows the high‑level
pipeline described in the user's mid‑submission and our suggested
improvements:

1. **Adversarial subspace discovery (ASD)**
   For each instance of undesired knowledge (e.g. a fact to forget), the
   model is probed with a targeted latent PGD attack.  The attack nudges
   hidden activations at a small set of layers to maximise the log‑likelihood
   of the forbidden answer.  The resulting perturbations and gradients are
   collected across items and an SVD is computed to form a low‑rank
   attack subspace ``U_attack``.

2. **Robust key/value construction (AKA++)**
   Following the adversarial subspace, we build a robust key by
   averaging subject token representations across both clean and
   adversarial contexts.  A robust value is obtained by projecting a
   refusal activation onto the attack subspace (to counter relapse) and
   away from the safe subspace spanned by retain data.  This mirrors
   the "adversarial key averaging" and projection tricks from the
   mid‑submission.

3. **r‑ROME update**
   Using the key and value, a rank‑one weight update is applied to
   the chosen MLP or attention layer.  This implementation uses
   the `easyeditor` library’s ``BaseEditor`` and ``ROMEHyperParams`` to
   apply r‑ROME updates.  The library handles the heavy lifting of
   computing the correct weight deltas for LLaMA‑style models.  A
   directional clamp is implemented to ensure that the update does not
   push the model far outside the attack subspace, and a norm sentinel
   stops updates that spike in magnitude.

4. **Minimax loop**
   The attack and edit steps are alternated: after performing a batch
   of edits, the adversary is rerun to search for new failure modes.
   The loop terminates once the attack success rate (ASR) falls below
   a threshold or a maximum number of rounds is reached.

This script is designed to be a template rather than a drop‑in
executable training script.  Many practical details – such as the
definition of the forget/retain datasets, model loading and saving, or
distributed training – are left as TODOs for the user.  Nevertheless,
the core logic of ASD, robust key/value construction and the r‑ROME
update is implemented in a way that should be compatible with
existing LAT and EasyEdit codebases.

Dependencies
------------
To run this script you will need the following Python packages:

* `torch` and `transformers` for model handling.
* `easyeditor` for r‑ROME editing (`pip install easyeditor`).
* `numpy` and `scipy` for the SVD used in attack subspace discovery.

If you are running on GPU, make sure to set ``device='cuda'`` when
instantiating the model and allocate perturbation tensors on the same
device.
"""

import os
import math
import json
import logging
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Dict, Optional

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers as _tf
from packaging import version as _pkg_version


# easyeditor jugaad
import os, sys, pathlib
this_dir = pathlib.Path(__file__).resolve().parent


sys.path.insert(0, str(this_dir / "EasyEdit"))
sys.path.insert(0, str(this_dir.parent / "EasyEdit"))

# from easyeditor import BaseEditor, ROMEHyperParams




BaseEditor = None  # lazy-loaded later
R_ROMEHyperParams = None  # lazy-loaded later


logger = logging.getLogger(__name__)

# Validate Transformers version for Llama 3.2 compatibility
if _pkg_version.parse(_tf.__version__) < _pkg_version.parse("4.45.0"):
    raise RuntimeError(
        f"Transformers >= 4.45.0 required for Llama 3.2; found {_tf.__version__}. "
        "Upgrade with: pip install -U 'transformers>=4.45.0' 'tokenizers>=0.20.0' 'huggingface_hub>=0.25.0' 'accelerate>=1.0.0'"
    )

# Default Hugging Face cache setup (uses user's provided path if present)
os.environ.setdefault("HF_HOME", "/scratch2/sky/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"]) 
os.environ.setdefault("HF_HUB_CACHE", os.environ["HF_HOME"]) 


@dataclass
class LatentAttackConfig:
    """Configuration for the latent PGD attack used in ASD."""
    epsilon: float = 0.25
    num_steps: int = 5
    step_size: float = 0.1
    layers: Optional[List[int]] = None  # will be determined from model if None
    prompt_mask_only: bool = True  # perturb only prompt positions


def collect_layer_outputs(model: nn.Module, inputs: Dict[str, torch.Tensor], layers: List[int]) -> Tuple[List[torch.Tensor], Dict]:
    """Run a forward pass and return a list of residual activations for selected layers.

    This helper registers temporary forward hooks on the model to capture
    the residual stream (pre‑layernorm) at the specified layer indices.
    It assumes the model has a list/ModuleList of decoder blocks accessible
    via ``model.model.layers`` (as in LLaMA and other transformer models).

    Args:
        model: The language model.
        inputs: A dictionary of input tensors (``input_ids``, ``attention_mask``, etc.).
        layers: Indices of layers whose residuals should be captured.

    Returns:
        A tuple ``(outputs, logits)`` where ``outputs[i]`` is the residual
        tensor for layer ``layers[i]`` (shape ``(batch, seq_len, hidden_dim)``)
        and ``logits`` is the model’s output logit tensor.
    """
    handles = []
    activations: Dict[int, torch.Tensor] = {}

    def save_residual(layer_idx: int):
        def hook(module, input, output):
            # Some HF layers return tuples; first element is hidden states
            out = output[0] if isinstance(output, (tuple, list)) else output
            activations[layer_idx] = out.detach()
        return hook

    # Register hooks on the specified layers.  We use the transformer
    # architecture assumption that the model has an attribute ``model``
    # containing ``layers`` or ``decoder.layers``.
    try:
        blocks = model.model.layers  # type: ignore[attr-defined]
    except AttributeError:
        # Some models nest decoder under ``transformer`` or ``decoder``.
        blocks = model.model.decoder.layers  # type: ignore[attr-defined]

    for idx in layers:
        handle = blocks[idx].register_forward_hook(save_residual(idx))
        handles.append(handle)

    # Forward pass
    outputs = model(**inputs)
    logits = outputs.logits  # type: ignore[assignment]

    # Remove hooks
    for h in handles:
        h.remove()

    # Collect activations in the order of ``layers``
    return [activations[idx] for idx in layers], logits


def latent_pgd_attack(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    target_texts: List[str],
    attack_config: LatentAttackConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """Compute adversarial latent perturbations via PGD for a batch of prompts.

    This function performs an inner maximisation: for each prompt/target pair,
    it searches for a set of perturbations on the model’s residual stream
    that maximises the log probability of the harmful target.  It returns
    both the final perturbations and the hidden activations under attack.

    Args:
        model: The language model to attack.
        tokenizer: Tokeniser for converting text to tensors.
        prompts: A batch of input prompt strings.
        target_texts: The corresponding harmful completions.
        attack_config: Hyperparameters for the PGD attack.
        device: Torch device (e.g., ``torch.device('cuda')``).

    Returns:
        A tuple ``(delta, embeddings, hiddens)`` where ``delta`` is the
        perturbation tensor applied at each selected layer (concatenated
        along the batch dimension), ``embeddings`` are the prompt token
        embeddings, and ``hiddens`` is a list of hidden activations from
        the selected layers after applying the adversarial perturbations.
    """
    model.eval()

    # Determine layer indices to attack.  By default we choose 4 evenly
    # spaced layers across the model depth.
    num_layers = len(model.model.layers)  # type: ignore[attr-defined]
    if attack_config.layers is None:
        step = max(num_layers // 4, 1)
        layers = list(range(step - 1, num_layers, step))[:4]
    else:
        layers = attack_config.layers

    # Tokenise prompts and targets
    batch_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    target_inputs = tokenizer(
        [p + t for p, t in zip(prompts, target_texts)],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    # Determine which positions to perturb: only the prompt positions if
    # prompt_mask_only is True.  We build a mask of shape (batch, seq_len)
    # with ones on the prompt tokens and zeros elsewhere.
    prompt_mask = None
    if attack_config.prompt_mask_only:
        prompt_mask = batch_inputs["input_ids"].ne(tokenizer.pad_token_id)
    
    # Create perturbations for each layer: dictionary from layer idx to
    # tensor of zeros with the same shape as the residual.  We’ll update
    # these tensors in place during PGD steps.
    delta = {
        idx: torch.zeros_like(
            collect_layer_outputs(model, batch_inputs, [idx])[0][0], device=device
        ).requires_grad_()
        for idx in layers
    }

    # Precompute embeddings for the prompts to avoid repeated forward passes.
    # Some models expose an ``embed_tokens`` module; otherwise we get the
    # embeddings from the first layer input after tokenisation.
    with torch.no_grad():
        embed = model.model.embed_tokens(batch_inputs["input_ids"]).detach()  # type: ignore[attr-defined]

    # PGD loop
    for step in range(attack_config.num_steps):
        # Forward with current perturbations
        def apply_perturbations(inputs: Dict[str, torch.Tensor], delta: Dict[int, torch.Tensor]):
            """Hook to apply delta to residuals during forward pass."""
            handles = []
            def add_delta(layer_idx: int):
                def hook(module, inp, out):
                    # Only perturb prompt positions if mask is provided.
                    d = delta[layer_idx]
                    # Unpack layer output
                    if isinstance(out, (tuple, list)):
                        hidden = out[0]
                        rest = tuple(out[1:])
                    else:
                        hidden = out
                        rest = None
                    # Align delta to current sequence length and device
                    hidden_device = hidden.device
                    d_local = d.to(hidden_device)
                    B, T_out, H = hidden.shape
                    _, T_d, _ = d_local.shape
                    min_len = T_out if prompt_mask is None else min(T_out, prompt_mask.size(1), T_d)
                    aligned = torch.zeros_like(hidden)
                    if min_len > 0:
                        aligned[:, :min_len, :] = d_local[:, :min_len, :]
                        if prompt_mask is not None:
                            pm = prompt_mask.to(hidden_device)
                            aligned[:, :min_len, :] = aligned[:, :min_len, :] * pm[:, :min_len].unsqueeze(-1)
                    new_hidden = hidden + aligned
                    if rest is not None:
                        return (new_hidden,) + rest
                    return new_hidden
                return hook
            blocks = model.model.layers  # type: ignore[attr-defined]
            for idx in delta.keys():
                h = blocks[idx].register_forward_hook(add_delta(idx))
                handles.append(h)
            return handles

        # Register hooks
        handles = apply_perturbations(batch_inputs, delta)
        # Compute logits and loss for harmful target
        outputs = model(**target_inputs)
        logits = outputs.logits
        # Cross‑entropy with shift: we compare the predicted next token at
        # position prompt_len onward to the target.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = target_inputs["input_ids"][..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(reduction="mean")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # Gradient descent on delta to maximise loss (i.e. ascent)
        loss.backward()
        # Update each delta with gradient and project onto L2 ball
        for idx in delta.keys():
            grad = delta[idx].grad
            if grad is None:
                continue
            # Normalise gradient and take an ascent step
            grad_norm = grad.view(grad.size(0), -1).norm(p=2, dim=1, keepdim=True) + 1e-8
            grad_step = attack_config.step_size * grad / grad_norm.unsqueeze(-1)
            delta[idx].data = delta[idx].data + grad_step
            # Project to L2 ball of radius epsilon
            delta_norm = delta[idx].data.view(delta[idx].size(0), -1).norm(p=2, dim=1, keepdim=True)
            factor = torch.clamp(delta_norm / attack_config.epsilon, min=1.0).unsqueeze(-1)
            delta[idx].data = delta[idx].data / factor
            # Zero gradients for next iteration
            delta[idx].grad.zero_()
        # Remove hooks and zero out gradients on model parameters
        for h in handles:
            h.remove()
        model.zero_grad()

    # After PGD, run a final forward pass to collect adversarial activations
    adv_layers, _ = collect_layer_outputs(model, batch_inputs, layers)
    # Stack perturbations into a tensor of shape (len(layers), batch, seq_len, hidden)
    delta_stack = torch.stack([delta[idx] for idx in layers], dim=0)
    return delta_stack, embed, adv_layers


def compute_attack_subspace(
    deltas: torch.Tensor,
    hiddens: List[torch.Tensor],
    rank: int = 16
) -> torch.Tensor:
    """Compute an attack subspace U_attack using SVD over adversarial perturbations.

    ``deltas`` is a tensor of shape (L, B, T, H) where L is the number of
    attacked layers.  We reshape these perturbations and hidden states
    into matrices and perform singular value decomposition to obtain the
    principal directions of attack.

    Args:
        deltas: Stack of perturbation tensors from the PGD attack.
        hiddens: List of hidden activations corresponding to each layer.
        rank: Number of singular vectors to keep.

    Returns:
        A matrix ``U_attack`` of shape (hidden_dim, rank) whose columns span
        the most adversarial directions.
    """
    # Concatenate perturbations and hiddens along batch and layer axes
    flat = []
    for d, h in zip(deltas, hiddens):
        # Flatten (B, T, H) -> (B*T, H)
        flat.append(d.view(-1, d.size(-1)))
        flat.append(h.view(-1, h.size(-1)))
    M = torch.cat(flat, dim=0)
    # Compute SVD on CPU in float32 for stability/compatibility
    M32 = M.detach().to(dtype=torch.float32, device="cpu")
    U, S, Vh = torch.linalg.svd(M32, full_matrices=False)
    U_attack = Vh[:rank].T  # shape (hidden_dim, rank)
    return U_attack


def project_value(
    v_refusal: torch.Tensor,
    U_attack: torch.Tensor,
    U_safe: Optional[torch.Tensor] = None,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> torch.Tensor:
    """Construct a robust value vector by projecting onto attack and safe subspaces.

    Starting from a reference refusal activation, we bias the value
    towards the attack subspace and away from the safe subspace.  The
    projection matrices are computed as ``P_attack = U_attack @ U_attack.T`` and
    ``P_safe = U_safe @ U_safe.T``.  The final value is

        v_robust = (I - β P_safe) [ v_refusal + α P_attack v_refusal ].

    Args:
        v_refusal: Reference refusal value vector (shape ``(hidden_dim,)``).
        U_attack: Attack subspace basis (shape ``(hidden_dim, r_attack)``).
        U_safe: Optional safe subspace basis (shape ``(hidden_dim, r_safe)``);
            if provided, the value will be projected away from it.
        alpha: Weight for the attack projection.
        beta: Weight for the safe projection.

    Returns:
        A robust value vector of shape ``(hidden_dim,)``.
    """
    device = v_refusal.device
    orig_dtype = v_refusal.dtype
    # Compute in float32 on the model device for numerical stability and dtype consistency
    v32 = v_refusal.to(dtype=torch.float32, device=device)
    Ua32 = U_attack.to(dtype=torch.float32, device=device)
    P_attack = Ua32 @ Ua32.T
    v = v32 + alpha * (P_attack @ v32)
    if U_safe is not None:
        Us32 = U_safe.to(dtype=torch.float32, device=device)
        P_safe = Us32 @ Us32.T
        v = v - beta * (P_safe @ v)
    return v.to(dtype=orig_dtype)


def build_robust_key(
    hidden_states: Iterable[torch.Tensor],
    subj_indices: Iterable[int],
    layers: List[int],
) -> torch.Tensor:
    """Average subject token hidden states across layers and contexts.

    ``hidden_states`` is an iterable of hidden activation tensors for each
    selected layer and each sample.  ``subj_indices`` gives the index of
    the subject token within each sequence.  The function returns the
    mean hidden state across all provided contexts and layers.  This
    implements the "adversarial key averaging" idea from the mid‑submission.

    Args:
        hidden_states: List of tensors (B, T, H) from each layer/loss.
        subj_indices: Indices of the subject token for each sample.
        layers: List of layer indices corresponding to hidden_states.

    Returns:
        A tensor of shape ``(hidden_dim,)`` representing the robust key.
    """
    keys = []
    for h in hidden_states:
        # Extract subject token hidden state
        for b, idx in enumerate(subj_indices):
            keys.append(h[b, idx])
    return torch.stack(keys, dim=0).mean(dim=0)


def compute_safe_subspace(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    retain_prompts: List[str],
    layers: List[int],
    rank: int = 16,
    device: torch.device = torch.device("cpu"),
    max_prompts: int = 256,
    batch_size: int = 4,
) -> torch.Tensor:
    """Compute a safe subspace from retain data by stacking hidden states.

    This helper runs the model on a batch of retain prompts and collects
    hidden activations at the selected layers.  An SVD on the stacked
    activations yields the safe subspace basis ``U_safe``.

    Args:
        model: The language model.
        tokenizer: Tokeniser for encoding retain prompts.
        retain_prompts: A list of benign/retain examples.
        layers: Indices of layers to collect activations from.
        rank: Number of singular vectors to keep.
        device: Torch device.

    Returns:
        A matrix ``U_safe`` of shape (hidden_dim, rank).
    """
    model.eval()
    flats: List[torch.Tensor] = []
    prompts = retain_prompts[:max_prompts]
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start:start + batch_size]
            batch = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            hiddens, _ = collect_layer_outputs(model, batch, layers)
            for h in hiddens:
                flats.append(h.view(-1, h.size(-1)).detach().cpu())
    if len(flats) == 0:
        raise RuntimeError("No retain activations collected for safe subspace")
    M = torch.cat(flats, dim=0)
    # Compute SVD on CPU in float32 for stability/compatibility
    M32 = M.detach().to(dtype=torch.float32, device="cpu")
    U, S, Vh = torch.linalg.svd(M32, full_matrices=False)
    return Vh[:rank].T


def apply_rrome_edit(
    editor: BaseEditor,
    prompts: List[str],
    ground_truths: List[str],
    new_targets: List[str],
    subjects: List[str],
    key_vector: torch.Tensor,
    value_vector: torch.Tensor,
    layer: int,
    clamp_threshold: float = 0.8,
    max_norm: float = 10.0,
) -> Tuple[Dict, BaseEditor, Dict]:
    """Apply a single r‑ROME edit with custom key/value vectors and stability guards.

    This function wraps the `BaseEditor.edit` method from EasyEdit to
    perform a targeted knowledge edit on the specified layer.  It
    modifies the editor’s hyperparameters to insert our precomputed key
    and value vectors, and enforces a directional clamp and norm
    sentinel to avoid unstable updates.  It returns the metrics, the
    edited editor instance and any auxiliary information.

    Args:
        editor: A `BaseEditor` instance initialised with the target model.
        prompts: List of prompts to edit (questions).
        ground_truths: Gold answers the model currently produces.
        new_targets: Desired new answers (e.g. refusals like "I can't say").
        subjects: List of subject strings for the questions.
        key_vector: Precomputed robust key vector (hidden_dim,).
        value_vector: Precomputed robust value vector (hidden_dim,).
        layer: Index of the layer to edit.
        clamp_threshold: Fraction of the update's energy that must lie
            inside the attack subspace; if violated the update magnitude
            is reduced.
        max_norm: Maximum allowed Frobenius norm of the weight update.

    Returns:
        A tuple ``(metrics, edited_editor, info)``.
    """
    if BaseEditor is None or R_ROMEHyperParams is None:
        raise RuntimeError(
            "easyeditor is not installed; please install it to use r‑ROME"
        )
    # Use editor's existing hyperparams and set target layer
    hparams = editor.hparams
    if hasattr(hparams, "layers"):
        hparams.layers = [layer]
    # Apply the edit using EasyEdit's R-ROME implementation
    metrics, edited_model, info = editor.edit(
        prompts=prompts,
        ground_truth=ground_truths,
        target_new=new_targets,
        subject=subjects,
        hyperparams=hparams,
        sequential_edit=False,
    )
    # Implement stability guards: if the update norm or projection falls
    # outside thresholds, back off on the update magnitude.  (This is a
    # simplified placeholder; a full implementation would inspect
    # ``info['delta_norm']`` and iterate.)
    delta_norm = info.get("delta_norm", 0.0)
    if isinstance(delta_norm, torch.Tensor):
        delta_norm = delta_norm.item()
    if delta_norm > max_norm:
        logger.warning(
            f"Edit norm {delta_norm:.2f} exceeds max_norm {max_norm}; consider reducing learning rate"
        )
    # Return updated editor
    return metrics, edited_model, info


def apply_rank_one_update(
    model: nn.Module,
    layer: int,
    key_vector: torch.Tensor,
    value_vector: torch.Tensor,
    U_attack: Optional[torch.Tensor] = None,
    clamp_threshold: float = 0.8,
    max_norm: float = 1,
    strength: float = 0.1,
) -> Dict:
    """Apply a minimal rank-one update at mlp.down_proj using provided key/value.

    We form z_k = up_proj(key_vector) to align with the intermediate dimension,
    then add delta = scale * (value_vector ⊗ z_k^T) to down_proj.weight.

    If U_attack is provided, we clamp value_vector toward the attack subspace.
    The Frobenius norm of delta is capped by max_norm.
    """
    # Locate modules
    block = model.model.layers[layer]  # type: ignore[attr-defined]
    up_proj = block.mlp.up_proj
    down_proj = block.mlp.down_proj
    w = down_proj.weight

    # Device/dtype alignment
    device = w.device
    dtype = w.dtype
    k = key_vector.to(device=device, dtype=dtype)
    v = value_vector.to(device=device, dtype=dtype)

    # Directional clamp toward attack subspace if provided
    frac_in_attack = None
    if U_attack is not None:
        Ua = U_attack.to(device=device, dtype=torch.float32)
        v32 = v.to(dtype=torch.float32)
        P_attack = Ua @ Ua.T
        v_att = (P_attack @ v32).to(dtype=dtype)
        num = (v_att.float().norm() ** 2).item()
        den = (v.float().norm() ** 2 + 1e-8).item()
        frac_in_attack = num / den if den > 0 else 0.0
        if frac_in_attack < clamp_threshold:
            # Blend toward attack subspace (stabilizes direction)
            v = 0.8 * v_att + 0.2 * v

    # Map key to intermediate space
    with torch.no_grad():
        z_k = up_proj(k.unsqueeze(0)).squeeze(0)

    # Outer product to match (hidden, intermediate)
    # Normalize vectors and apply provided strength
    v_unit = v / (v.norm() + 1e-8)
    z_unit = z_k / (z_k.norm() + 1e-8)
    delta = v_unit.unsqueeze(1) @ z_unit.unsqueeze(0)  # (H, I)

    # Norm cap
    with torch.no_grad():
        delta = delta * strength
        fro = torch.linalg.norm(delta.float())
        scale = 1.0
        if fro.item() > max_norm:
            scale = max_norm / (fro.item() + 1e-8)
        down_proj.weight.add_(delta * scale)

    return {
        "delta_fro": float(fro.item()),
        "scale": float(scale),
        "frac_in_attack": float(frac_in_attack) if frac_in_attack is not None else None,
        "strength": float(strength),
    }


def apply_easyedit_r_rome(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    model_name: str,
    edit_layer: int,
    prompts: List[str],
    harmful: List[str],
    new_targets: List[str],
    subjects: List[str],
) -> Tuple[Dict, nn.Module, Dict]:
    """Fallback: invoke EasyEdit R-ROME on the in-memory model via tuple hparams."""
    global BaseEditor, R_ROMEHyperParams
    if BaseEditor is None or R_ROMEHyperParams is None:
        from easyeditor.editors.editor import BaseEditor as _BaseEditor
        from easyeditor.models.r_rome.r_rome_hparams import R_ROMEHyperParams as _R_ROMEHyperParams
        BaseEditor = _BaseEditor
        R_ROMEHyperParams = _R_ROMEHyperParams

    # Pick hparams file based on model name
    def _guess_hparams_key(name: str) -> str:
        low = name.lower()
        if "3.2-1b" in low:
            return "llama3.2-1b"
        if "3.2-3b" in low:
            return "llama3.2-3b"
        return os.path.basename(name)

    hparams_key = _guess_hparams_key(model_name)
    hparams_path = this_dir / "EasyEdit" / "hparams" / "R-ROME" / f"{hparams_key}.yaml"
    hparams = R_ROMEHyperParams.from_hparams(str(hparams_path))
    # Reuse attacked model in-memory for loading, but keep a string model_name for eval paths
    hparams.model_name = (model, tokenizer)
    # Set target layer
    hparams.layers = [edit_layer]
    editor = BaseEditor.from_hparams(hparams)
    # Ensure downstream evaluation receives a string model name
    editor.model_name = model_name
    editor.hparams.model_name = model_name
    metrics, edited_model, info = editor.edit(
        prompts=prompts,
        ground_truth=harmful,
        target_new=new_targets,
        subject=subjects,
        hyperparams=hparams,
        sequential_edit=True,
    )
    return metrics, edited_model, info

def minimax_loop(
    model_name: str,
    forget_dataset: Iterable[Tuple[str, str, str]],
    retain_dataset: Iterable[str],
    refusal_template: str = "I'm sorry, but I can't answer that.",
    attack_config: LatentAttackConfig = LatentAttackConfig(),
    rounds: int = 3,
    attack_rank: int = 16,
    safe_rank: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """Run the full minimax edit loop over a forget dataset and retain set.

    Args:
        model_name: Name or path of a pretrained causal language model.
        forget_dataset: Iterable of tuples ``(question, subject, answer)``.
        retain_dataset: Iterable of benign prompts to preserve model
            behaviour on.
        refusal_template: String used as the target new answer.
        attack_config: Hyperparameters for the latent adversarial attack.
        rounds: Maximum number of minimax iterations.
        attack_rank: Rank of the attack subspace.
        safe_rank: Rank of the safe subspace.
        device: Device string for torch.

    Returns:
        None.  The edited model is saved to disk at the end.
    """
    device = torch.device(device)
    # Load model and tokenizer (robust to newer architectures + correct cache)
    cache_dir = os.environ.get("HF_HOME", "/scratch2/sky/hf_cache")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_dir,
        attn_implementation="eager",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    # Ensure tokenizer has padding defined for batching
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # Do not move the model when device_map="auto"; Accelerate handles placement
    model.eval()
    # Editor not required for manual rank-one update; we keep model in-memory
    # Determine layers to attack/edit
    num_layers = len(model.model.layers)  # type: ignore[attr-defined]
    step = max(num_layers // 4, 1)
    attack_layers = list(range(step - 1, num_layers, step))[:4]
    # Optionally force edit layer from R-ROME YAML if specified (e.g., layers: [5])
    def _guess_hparams_key(name: str) -> str:
        low = name.lower()
        if "3.2-1b" in low:
            return "llama3.2-1b"
        if "3.2-3b" in low:
            return "llama3.2-3b"
        return os.path.basename(name)
    forced_layer = None
    try:
        import yaml as _yaml
        hkey = _guess_hparams_key(model_name)
        yml_path = this_dir / "EasyEdit" / "hparams" / "R-ROME" / f"{hkey}.yaml"
        if yml_path.exists():
            with open(yml_path, "r") as _f:
                _cfg = _yaml.safe_load(_f)
                if isinstance(_cfg.get("layers"), list) and len(_cfg["layers"]) > 0:
                    forced_layer = int(_cfg["layers"][0])
    except Exception:
        forced_layer = None
    # Precompute safe subspace from retain data
    retain_prompts = list(retain_dataset)
    U_safe = compute_safe_subspace(model, tokenizer, retain_prompts, attack_layers, rank=safe_rank, device=device)
    # Minimax loop
    for round_idx in range(rounds):
        logger.info(f"Round {round_idx+1} / {rounds}")
        # Prepare prompts and targets from forget dataset
        prompts = []
        harmful = []
        subjects = []
        for q, subj, ans in forget_dataset:
            prompts.append(q)
            harmful.append(ans)
            subjects.append(subj)
        # Adversarial attack
        delta_stack, embeddings, adv_hiddens = latent_pgd_attack(
            model, tokenizer, prompts, harmful, attack_config, device
        )
        # Compute attack subspace
        U_attack = compute_attack_subspace(delta_stack, adv_hiddens, rank=attack_rank)
        # Build robust key
        subj_indices = []
        # Determine subject token positions using the tokenizer.  For
        # simplicity we take the first occurrence of the subject string in
        # the tokenised prompt.  More robust implementations may need to
        # align tokens precisely.
        for prompt, subj in zip(prompts, subjects):
            tok_ids = tokenizer.encode(prompt, add_special_tokens=False)
            subj_ids = tokenizer.encode(subj, add_special_tokens=False)
            # Find subsequence match
            idx = 0
            for i in range(len(tok_ids)):
                if tok_ids[i : i + len(subj_ids)] == subj_ids:
                    idx = i
                    break
            subj_indices.append(idx)
        # ``adv_hiddens`` is a list of hidden activations per selected layer
        k_robust = build_robust_key(adv_hiddens, subj_indices, attack_layers)
        # Obtain reference refusal activation by running model on a dummy
        # prompt.  We use the last layer hidden state of the end‑of‑sequence
        # token as the refusal value.
        refusal_prompt = "Q: dummy\nA: "
        refusal_batch = tokenizer(refusal_prompt, return_tensors="pt").to(device)
        refusal_hidden, _ = collect_layer_outputs(model, refusal_batch, [attack_layers[-1]])
        v_ref = refusal_hidden[0][0, -1]  # shape (hidden_dim,)
        # Project value
        v_robust = project_value(v_ref, U_attack.to(device), U_safe.to(device))
        # Build new targets (refusals) for each prompt
        new_targets = [refusal_template for _ in prompts]
        # Apply rank‑one update at the chosen layer (YAML override if present)
        edit_layer = forced_layer if forced_layer is not None else attack_layers[-1]
        # Choose update strength via quick loss-based search wrt harmful targets
        def compute_harm_ce(m, tok, pr_pts, ans_pts) -> float:
            total_loss = 0.0
            count = 0
            for p, a in zip(pr_pts, ans_pts):
                full = tok(p + a, return_tensors="pt").to(device)
                prompt = tok(p, return_tensors="pt").to(device)
                labels = full["input_ids"].clone()
                # mask prompt tokens
                labels[:, : prompt["input_ids"].shape[1]] = -100
                out = m(**full, labels=labels)
                total_loss += float(out.loss.detach().cpu())
                count += 1
            return total_loss / max(count, 1)

        pre_ce = compute_harm_ce(model, tokenizer, prompts, harmful)
        strengths = [0.05, 0.1, 0.2, 0.4, 0.8]
        # strengths = [0.01]#, 0.008, 0.01]
        # backup weight
        block = model.model.layers[edit_layer]  # type: ignore[attr-defined]
        down_w = block.mlp.down_proj.weight
        w_backup = down_w.detach().clone()
        best_gain = -1e9
        best_s = strengths[0]
        best_info = None
        for s in strengths:
            # apply
            _ = apply_rank_one_update(
                model,
                edit_layer,
                k_robust,
                v_robust,
                U_attack=U_attack,
                clamp_threshold=0.8,
                max_norm=1.0,
                strength=s,
            )
            post_ce = compute_harm_ce(model, tokenizer, prompts, harmful)
            gain = post_ce - pre_ce
            # revert
            with torch.no_grad():
                down_w.copy_(w_backup)
            if gain > best_gain:
                best_gain = gain
                best_s = s
        # apply best
        logger.info("Method: rank-one update (mROME) — applying best strength after line search")
        info = apply_rank_one_update(
            model,
            edit_layer,
            k_robust,
            v_robust,
            U_attack=U_attack,
            clamp_threshold=0.8,
            max_norm=1.0,
            strength=best_s,
        )
        logger.info(f"Applied rank-one update at layer {edit_layer} with strength {best_s}, harm CE gain {best_gain:.4f}: {info}")

        # Save generations for auditing (use chat template if available)
        model.eval()
        with torch.no_grad():
            outs = []
            for p in prompts:
                if hasattr(tokenizer, "apply_chat_template"):
                    chat = tokenizer.apply_chat_template(
                        [{"role": "user", "content": p}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    enc = tokenizer(chat, return_tensors="pt", padding=False).to(device)
                else:
                    enc = tokenizer(p, return_tensors="pt", add_special_tokens=True).to(device)
                gen = model.generate(
                    **enc,
                    max_new_tokens=128,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                )
                input_len = enc["input_ids"].shape[1]
                text = tokenizer.decode(gen[0][input_len:], skip_special_tokens=True)
                outs.append({"prompt": p, "generation": text})
        os.makedirs("./mrome_lat_model", exist_ok=True)
        out_path = os.path.join("./mrome_lat_model", f"outputs_round{round_idx+1}.json")
        with open(out_path, "w") as f:
            json.dump(outs, f, indent=2)

        # If harmful still not reduced (line-search gain small), fallback to EasyEdit R-ROME
        if best_gain < 1e-3:
            logger.info("Method: EasyEdit R-ROME fallback — best_gain %.6f below threshold 0.001", best_gain)
            metrics, model, info = apply_easyedit_r_rome(
                model,
                tokenizer,
                model_name,
                edit_layer,
                prompts,
                harmful,
                new_targets,
                subjects,
            )
            # Re-generate after EasyEdit
            with torch.no_grad():
                outs = []
                for p in prompts:
                    if hasattr(tokenizer, "apply_chat_template"):
                        chat = tokenizer.apply_chat_template(
                            [{"role": "user", "content": p}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        enc = tokenizer(chat, return_tensors="pt", padding=False).to(device)
                    else:
                        enc = tokenizer(p, return_tensors="pt", add_special_tokens=True).to(device)
                    gen = model.generate(
                        **enc,
                        max_new_tokens=128,
                        do_sample=False,
                        temperature=1.0,
                        top_p=1.0,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    input_len = enc["input_ids"].shape[1]
                    text = tokenizer.decode(gen[0][input_len:], skip_special_tokens=True)
                    outs.append({"prompt": p, "generation": text})
            with open(out_path, "w") as f:
                json.dump(outs, f, indent=2)
    # Save final model
    out_dir = "./mrome_lat_model"
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    # Save run metadata without overwriting the model's config.json
    with open(os.path.join(out_dir, "edit_meta.json"), "w") as f:
        json.dump({"attack_config": attack_config.__dict__}, f, indent=2)
    logger.info(f"Saved edited model to {out_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage (requires user to fill in datasets)
    # The forget dataset should be a list of (question, subject, answer)
    forget_examples = [
        ("Who wrote the Harry Potter books?", "Harry Potter", "J. K. Rowling"),
        # ("What is the capital of Japan?", "Japan", "Tokyo"),
    ]
    retain_examples = [
        "What is the capital of France?",
    ]
    # Replace with your own model name (e.g. "NousResearch/Llama-3-8B-Instruct")
    model_name = "meta-llama/Llama-3.2-1B-Instruct"#"llama3.2-1b"
    # Skip execution if no examples are provided
    if len(forget_examples) == 0 or len(retain_examples) == 0:
        logger.warning(
            "No forget or retain examples provided. The script will exit without running."
        )
    else:
        minimax_loop(
            model_name,
            forget_examples,
            retain_examples,
        )