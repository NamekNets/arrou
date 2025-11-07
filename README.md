Minimax r‑ROME + LAT (mROMΕ) – Llama‑3.2 Unlearning
===================================================

This repo contains a proof‑of‑concept pipeline for attacker‑in‑the‑loop editing: a targeted latent attacker (LAT) discovers adversarial directions while robust R‑ROME edits apply guarded, layer‑local rank‑one updates to remove specific knowledge with minimal collateral damage.

What’s here (most useful first)
--------------------------------
- mrome_lat_integration.py
  - End‑to‑end POC of the minimax loop:
    - runs latent PGD to find an attack subspace (U_attack)
    - builds a safe subspace from retain data (U_safe)
    - constructs a refusal‑projected value
    - applies a small, rank‑one update (mROME) with a quick line‑search
    - if the update is ineffective, falls back to EasyEdit’s R‑ROME on the same in‑memory model
  - Saves edited weights and per‑round generations to mrome_lat_model/.
- tofu_experiment.py
  - Runs the pipeline on TOFU‑style data (JSONL or a TOFU split directory).
  - Optional subject normalization with a clean Llama to build (question, subject, answer) tuples.
  - Computes and saves metrics (forget QA probability, ROUGE, refusal rate, retain ROUGE).
- tofu_experiment.sh
  - Example shell wrapper showing how to call tofu_experiment.py with sensible defaults.
- quick_infer.py
  - Minimal script to load the edited model and test one prompt (or compare side‑by‑side with the base repo id).
- ideas.md
  - Short design note explaining the LAT→subspace→r‑ROME concept. Use it as background; the code implements a simplified/robust version.

Environment setup
-----------------
- Python 3.10 recommended.
- PyTorch >= 2.1 (2.3.x works well).
- Transformers >= 4.45.0 and Tokenizers >= 0.20.0 for Llama‑3.2.
- Local EasyEdit checkout is expected (this repo includes EasyEdit/); we import only the needed modules to avoid heavy optional deps.
- Use a persistent HF cache (the code defaults to HF_HOME=/scratch2/sky/hf_cache). This prevents re‑downloads and avoids shape‑mismatch issues across envs.

Example:
```bash
conda create -n llama32 python=3.10 -y
conda activate llama32
pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
pip install 'transformers>=4.45.0' 'tokenizers>=0.20.0' 'huggingface_hub>=0.25.0' 'accelerate>=1.0.0' safetensors sentencepiece packaging
export HF_HOME=/scratch2/sky/hf_cache
```

How to run
----------
1) Quick sanity check with the POC loop
- Edit the small, in‑file examples in mrome_lat_integration.py (at the bottom).
- Run:
```bash
CUDA_VISIBLE_DEVICES=0 python mrome_lat_integration.py
```
Outputs:
- mrome_lat_model/
  - outputs_round{n}.json and outputs_round{n}_clean.json (per‑round generations)
  - metrics_round{n}.json (from tofu_experiment.py only)
  - edit_meta.json (run metadata)
  - model.safetensors and tokenizer files (final edited model)

2) One‑off qualitative check
- quick_infer.py loads edited weights and prints a single prompt response.
- If your edited dir only contains model.safetensors (no valid config.json), pass --base_model so it can load config from the original repo.
```bash
CUDA_VISIBLE_DEVICES=0 python quick_infer.py --model_dir ./mrome_lat_model --base_model meta-llama/Llama-3.2-1B-Instruct
```
- Or edit PROMPT in quick_infer.py and run:
```bash
CUDA_VISIBLE_DEVICES=0 python quick_infer.py
```

3) TOFU experiments and metrics
- Place TOFU JSON files (e.g., forget01.json, retain99.json) under a directory.
- Run tofu_experiment.py by pointing to the directory (auto‑picks splits) or to explicit JSON files:
```bash
# directory auto-pick (prefer forget01.json, retain99/95/90.json if present)
CUDA_VISIBLE_DEVICES=0 python tofu_experiment.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --forget_jsonl ./data/tofu/<your_split_dir> \
  --limit 40 --rounds 3 \
  --normalize_subjects

# or explicit files
CUDA_VISIBLE_DEVICES=0 python tofu_experiment.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --forget_jsonl ./data/tofu/.../forget01.json \
  --retain_jsonl ./data/tofu/.../retain99.json \
  --limit 40 --rounds 3 --normalize_subjects
```
What it does:
- Loads forget (question+answer) and optional retain prompts.
- If --normalize_subjects, uses a clean Llama to produce verbatim‑in‑question subjects and builds (question, subject, answer) tuples.
- Runs the minimax loop and saves:
  - per‑round generations (outputs_round*.json and outputs_round*_clean.json)
  - final metrics (metrics_round{n}.json):
    - forget: refusal_rate, avg_answer_prob (geomean token prob of harmful answer), avg_rougeL
    - retain: avg_rougeL (if retain has answers)

Notes about mrome_lat_integration.py
------------------------------------
- Attacker (LAT): latent PGD perturbs a few evenly‑spaced layers to maximize harmful likelihood and collects hiddens/perturbations → SVD → U_attack.
- Safe subspace: runs the model on retain prompts (batched and CPU‑accumulated) → SVD → U_safe.
- Value: refusal activation projected toward U_attack and away from U_safe (float32 math, then cast back).
- Edit: small, rank‑one update at a decisive layer (default from YAML; otherwise last attack layer). A quick line‑search picks a safe strength; the delta is norm‑capped.
- Fallback: if the strength search gives negligible forgetting (best_gain < 1e‑3), we call EasyEdit’s R‑ROME on the same in‑memory model.
- Generation: always uses the chat template (when available) and slices off the prompt by input length before decoding.
- HF cache is never overwritten; we save the edited model to mrome_lat_model/.

tofu_experiment.sh
------------------
- A small shell wrapper you can customize to run tofu_experiment.py with your dataset paths, rounds, and subject normalization flags.

Common pitfalls & fixes
-----------------------
- Shape mismatch (e.g., [512, 2048] vs [2048, 2048]): use Transformers >= 4.45.0 and PyTorch >= 2.1 for Llama‑3.2 (GQA support).
- “Gibberish” outputs: ensure chat template + greedy decoding + correct prompt slicing; avoid compounding large edits; align edit_layer == v_loss_layer.
- Subject inconsistency: unlearning is stronger when all items for an entity share an identical subject string that appears verbatim in the question (use --normalize_subjects).
- OOM during safe subspace: the code already streams retain prompts in mini‑batches and moves activations to CPU; reduce max_prompts/batch_size if needed.


Links for dataset and weights: https://drive.google.com/drive/folders/1lt_u8rwMVSebGes_vM_pBM8JXmcbfrza?usp=sharing


FAQ
---
- Does this overwrite my HF cache?
  - No. The edited model is saved under mrome_lat_model/ (or your chosen OUT_DIR); the cache is read‑only.
- Where do I change which layer to edit?
  - YAML under EasyEdit/hparams/R-ROME/ (e.g., llama3.2-1b.yaml) has layers: [L] and v_loss_layer: L. The code aligns to those when present.
- Can I try a single, simple R‑ROME run?
  - Use mrome_simple_experiment.py. It loads the model, applies EasyEdit’s R‑ROME to your small tuple list, and saves generations to ./mrome_simple_model/.

License
-------
Research prototype; see upstream component licenses for EasyEdit, Transformers, and datasets.
