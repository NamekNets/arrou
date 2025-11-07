"""
Training script for ARROU (Adversarially Robust Rank‑One Unlearning).

This script demonstrates how to apply the ARROU methodology described in
the mid‑project report to a collection of prompts requiring unlearning. It
leverages the utility functions implemented in ``arro_methodology.py`` to
automate the three core phases of ARROU:

1. **Layer discovery via LAT** – use a small sample of data to rank
   transformer layers by how much they store the target knowledge
   according to attack success rate【185935615541641†L330-L361】.
2. **Stable unlearning via r‑ROME** – compute key and value vectors and
   apply a rank–one edit to redirect the model from the original answer
   toward a refusal【185935615541641†L372-L388】.
3. **Adversarial robustness enhancement** – optionally recompute the key
   vector using adversarial contexts and project the value vector away from
   attack directions【185935615541641†L444-L466】.

This script assumes you have placed ``arro_methodology.py`` on your
``PYTHONPATH`` (for example, by adding it to the ``latent_at`` package
within ``lat-train``). It also expects access to a set of forget prompts and
an optional retain/forget dataset for layer discovery. The forget prompts
can be provided via a JSON file or listed directly on the command line.

Example usage:

.. code-block:: bash

   python train_arrou.py \
       --model_name meta-llama/Llama-3.2-1B-Instruct \
       --forget_prompts_json /mnt/Shared-Storage/sid/datasets/wmdp-corpora/bio-forget-corpus.jsonl \
       --candidate_layers 4 8 12 16 \
       --refusal_text "I don't know" \
       --output_dir saves/arrou_model

.. note::
   For realistic training you should replace the toy PGD adversary used
   here with the more sophisticated LAT implementation from the
   ``latent_at.lat_methods.ProjectedGradLAT`` class, and supply actual
   datasets for both forget and retain corpora.

"""

import argparse
import json
import os
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from arrou_methodology import (
    discover_layers,
    compute_key_vector,
    compute_value_vector,
    robust_unlearning,
)

# Optional: import WMDP dataset utilities if available. If not available
# (e.g., outside of the ``lat-train`` repo), the script will fall back to
# using only the forget prompts provided.
try:
    from latent_at.lat_datasets import load_targeted_wmdp_data, make_targeted_wmdp_dataloader
    LAT_DATA_AVAILABLE = True
except ImportError:
    LAT_DATA_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model with ARROU unlearning.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name or path of the pretrained model (HuggingFace identifier).",
    )
    parser.add_argument(
        "--forget_prompts_json",
        type=str,
        default=None,
        help=(
            "Path to a file containing prompts requiring unlearning. "
            "The file may be a JSON array of strings (.json) or a JSON Lines file (.jsonl) "
            "where each line is a JSON object with a 'text' field."
        ),
    )
    parser.add_argument(
        "--forget_prompts",
        type=str,
        nargs="*",
        default=None,
        help="Prompts requiring unlearning (use instead of --forget_prompts_json).",
    )
    parser.add_argument(
        "--candidate_layers",
        type=int,
        nargs="*",
        default=[4, 8, 12, 16, 20, 24],
        help="Candidate transformer layers to probe for storing the target knowledge.",
    )
    parser.add_argument(
        "--refusal_text",
        type=str,
        default="I don't know",
        help="Text that the model should output after unlearning.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=2.0,
        help="L∞ norm bound for the PGD adversary during layer discovery and adversarial context generation.",
    )
    parser.add_argument(
        "--pgd_lr",
        type=float,
        default=5e-2,
        help="Learning rate for the PGD adversary.",
    )
    parser.add_argument(
        "--pgd_iters",
        type=int,
        default=16,
        help="Number of PGD iterations during layer discovery and adversarial context generation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training (cuda or cpu).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the edited model and tokenizer will be saved.",
    )
    parser.add_argument(
        "--use_dataset",
        action="store_true",
        help="Use the WMDP forget/retain dataset for layer discovery. Requires latent_at to be installed.",
    )
    parser.add_argument(
        "--reduce_prefixes",
        type=int,
        default=10,
        help="Number of random prefixes to use in key vector computation (lower = less memory).",
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        default=True,
        help="Use float16 precision to reduce memory usage.",
    )
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Maximum number of prompts to process (useful for testing with limited memory).",
    )
    return parser.parse_args()


def load_prompts(path: str) -> List[str]:
    """
    Load a list of forget prompts from either a JSON array or a JSONL file.

    If ``path`` ends with ``.jsonl``, each line is parsed as a JSON object and
    the ``text`` field is extracted if present (this mirrors the structure of
    the WMDP corpora used in ``lat-train``). Otherwise the file is assumed
    to contain a JSON array of strings. This helper returns a plain list of
    prompt strings suitable for unlearning.

    Args:
        path: Path to the prompt file, either ``.json`` or ``.jsonl``.

    Returns:
        A list of prompt strings.

    Raises:
        ValueError: If the JSON file does not contain a list of strings or the
            JSONL file contains objects without a ``text`` field.
    """
    if path.endswith(".jsonl"):
        prompts: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                # Expect a 'text' field in each record; fall back to the entire object
                if isinstance(record, dict):
                    if "text" in record:
                        prompts.append(str(record["text"]))
                    else:
                        # If there is no 'text' key, use the first string value found
                        # or the repr of the entire record
                        first_string = next((v for v in record.values() if isinstance(v, str)), None)
                        if first_string is not None:
                            prompts.append(first_string)
                        else:
                            raise ValueError(
                                f"Line {lineno} in {path} does not contain a 'text' field or any string values."
                            )
                else:
                    # Non-dict lines are treated as raw prompt strings
                    prompts.append(str(record))
        if not prompts:
            raise ValueError(f"No prompts were found in {path}; ensure it contains JSON objects per line.")
        return prompts
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(item) for item in data]
        raise ValueError("The JSON file must contain a list of strings.")


def get_layer_discovery_dataloader(
    tokenizer: AutoTokenizer, use_dataset: bool
) -> Optional[DataLoader]:
    """If requested and available, construct a dataloader for layer discovery using WMDP data."""
    if use_dataset and LAT_DATA_AVAILABLE:
        # By default, probe on WMDP cyber forget with WikiText retain; adjust as needed.
        datasets = load_targeted_wmdp_data(retain_corpora=["wikitext"], forget_corpora=["cyber-forget-corpus"])
        dataloader = make_targeted_wmdp_dataloader(datasets[0], tokenizer, lat_batch_size=2, data_truncate_length=256)
        return dataloader
    return None


def generate_adversarial_context(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    layer: int,
    epsilon: float,
    lr: float,
    iters: int,
    device: str,
) -> Tuple[str, torch.Tensor]:
    """
    Generate a single adversarial context and perturbation for the given prompt
    using PGD. This helper closely mirrors the adversary in
    ``arro_methodology._pgd_attack`` but is defined locally for clarity.

    Returns a tuple ``(context_str, delta)`` where ``context_str`` is the
    original prompt and ``delta`` is the learned perturbation tensor.
    """
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    # Get model's dtype to ensure consistency
    model_dtype = next(model.parameters()).dtype
    
    perturb = torch.zeros(1, 1, model.config.hidden_size, device=device, dtype=model_dtype, requires_grad=True)

    def hook_fn(module, inp, out):
        return out + perturb

    # Attach hook to the specified layer's MLP
    handle = model.model.layers[layer].mlp.register_forward_hook(hook_fn)
    optim = torch.optim.Adam([perturb], lr=lr)
    # Use a simple negative log probability loss of the original output as target
    with torch.no_grad():
        base_output = model(input_ids=input_ids).logits[:, -1]
        base_label = base_output.argmax(dim=-1)
    for _ in range(iters):
        optim.zero_grad()
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, -1]
        loss = -torch.nn.functional.log_softmax(logits, dim=-1)[0, base_label]
        (-loss).backward()
        optim.step()
        with torch.no_grad():
            perturb.data = torch.clamp(perturb.data, -epsilon, epsilon)
    handle.remove()
    return prompt, perturb.detach().squeeze(0)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Memory optimization: Load model with reduced precision and memory-efficient settings
    print("Loading model with memory optimizations...")
    load_kwargs = {
        "low_cpu_mem_usage": True,
        "device_map": args.device,
    }
    if args.use_fp16:
        load_kwargs["torch_dtype"] = torch.float16
        print("Using float16 precision")
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Enable gradient checkpointing if available
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    # Load forget prompts
    if args.forget_prompts_json:
        forget_prompts = load_prompts(args.forget_prompts_json)
    elif args.forget_prompts:
        forget_prompts = list(args.forget_prompts)
    else:
        raise ValueError("Either --forget_prompts_json or --forget_prompts must be provided.")
    
    # Limit prompts if max_prompts is specified
    if args.max_prompts is not None:
        forget_prompts = forget_prompts[:args.max_prompts]
        print(f"Processing only {len(forget_prompts)} prompts (max_prompts={args.max_prompts})")

    # Optionally build dataloader for layer discovery
    dataloader = get_layer_discovery_dataloader(tokenizer, args.use_dataset)
    # Discover the most critical layer to edit
    if dataloader is not None:
        ranked_layers = discover_layers(
            model=model,
            dataloader=dataloader,
            candidate_layers=args.candidate_layers,
            epsilon=args.epsilon,
            pgd_lr=args.pgd_lr,
            pgd_iters=args.pgd_iters,
            device=args.device,
        )
        target_layer = ranked_layers[0]
        print(f"Selected layer {target_layer} for unlearning based on ASR ranking.")
    else:
        # If no dataset is available, pick the first candidate layer
        target_layer = args.candidate_layers[0]
        print(f"No layer discovery dataset provided; defaulting to layer {target_layer}.")

    # Iterate over each prompt to unlearn
    for idx, prompt in enumerate(forget_prompts):
        print(f"\n[{idx+1}/{len(forget_prompts)}] Unlearning prompt: {prompt}")
        
        # Clear GPU cache before processing each prompt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Compute and apply a robust r‑ROME edit
        # 1. Compute adversarial context (optional – here we generate one adversarial perturbation)
        adv_contexts: List[Tuple[str, torch.Tensor]] = []
        try:
            # Generate an adversarial perturbation for the prompt on the discovered layer
            adv_ctx = generate_adversarial_context(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                layer=target_layer,
                epsilon=args.epsilon,
                lr=args.pgd_lr,
                iters=args.pgd_iters,
                device=args.device,
            )
            adv_contexts.append(adv_ctx)
            print("Generated adversarial context for robust key recomputation.")
        except Exception as e:
            print(f"[warn] Failed to generate adversarial context: {e}. Proceeding without.")
            adv_contexts = []
            # Clear cache after failed attempt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 2. Apply robust unlearning
        robust_unlearning(
            model=model,
            tokenizer=tokenizer,
            subject_prompt=prompt,
            refusal_text=args.refusal_text,
            layer=target_layer,
            adversarial_contexts=adv_contexts,
            alpha=0.5,
            num_prefixes=args.reduce_prefixes,
            device=args.device,
        )
        print("Applied robust unlearning edit.")
        
        # Clear cache after each prompt to prevent memory accumulation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save the edited model and tokenizer
    print(f"\nSaving edited model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete.")


if __name__ == "__main__":
    main()