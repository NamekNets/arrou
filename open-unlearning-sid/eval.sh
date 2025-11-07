#!/bin/bash
set -euo pipefail

method="${1:-ga}" # ga or rmu
unlearning_task="${2:-bio}" # bio or cyber

MODEL_NAME=Llama-3.2-1B-Instruct
BASE_DIR=/mnt/Shared-Storage/HF_CACHE/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6
ADAPTER_DIR=/home/sid/nameknets/unlearning_wmdp_${unlearning_task}_ga_test_save
RMU_MODEL_DIR=/home/sid/nameknets/unlearning_wmdp_${unlearning_task}_rmu_test_save

LOG_DIR=saves/eval/WMDP_${unlearning_task^^}_LAT_${method^^}
mkdir -p "${LOG_DIR}"

echo "activating environment"
set --  # avoid passing script args into 'activate'
source /mnt/Shared-Storage/sid/miniconda3/bin/activate
conda activate nameknets

export HYDRA_FULL_ERROR=1
# export TRANSFORMERS_OFFLINE=1
# export HF_HUB_OFFLINE=1

#############################################
# Helper: ensure MODEL_DIR has a fast tokenizer
#############################################
ensure_tokenizer_in_dir() {
  local target_dir="$1"
  local source_dir="$2"

  # If target already has a tokenizer, do nothing
  if [[ -f "${target_dir}/tokenizer.json" || -f "${target_dir}/tokenizer.model" ]]; then
    echo "[TOK] ${target_dir} already has tokenizer assets."
    return 0
  fi

  # Build a fast tokenizer bundle (tokenizer.json) from source_dir if missing there
  local fast_src="${source_dir}"
  if [[ ! -f "${source_dir}/tokenizer.json" ]]; then
    echo "[TOK] ${source_dir} lacks tokenizer.json; building a fast bundle..."
    local FAST_TOK_DIR="/mnt/Shared-Storage/sid/checkpoints/tokenizers/${MODEL_NAME}_fast"
    if [[ ! -f "${FAST_TOK_DIR}/tokenizer.json" ]]; then
      mkdir -p "${FAST_TOK_DIR}"
      BASE_DIR="${source_dir}" FAST_TOK_DIR="${FAST_TOK_DIR}" python - <<'PY'
from transformers import AutoTokenizer
import os, shutil, sys
base = os.environ["BASE_DIR"]; out = os.environ["FAST_TOK_DIR"]
try:
    tok = AutoTokenizer.from_pretrained(base, use_fast=True, local_files_only=True, legacy=False)
except Exception:
    tok = AutoTokenizer.from_pretrained(base, use_fast=False, local_files_only=True)
tok.save_pretrained(out)
for fn in ("tokenizer_config.json","special_tokens_map.json"):
    src = os.path.join(base, fn)
    if os.path.exists(src): shutil.copy2(src, out)
print("Saved fast tokenizer to", out)
PY
    fi
    fast_src="${FAST_TOK_DIR}"
  fi

  echo "[TOK] Copying tokenizer files from ${fast_src} -> ${target_dir}"
  mkdir -p "${target_dir}"
  cp -f "${fast_src}/tokenizer.json" "${target_dir}/" 2>/dev/null || true
  cp -f "${fast_src}/tokenizer_config.json" "${target_dir}/" 2>/dev/null || true
  cp -f "${fast_src}/special_tokens_map.json" "${target_dir}/" 2>/dev/null || true
}

#############################################
# Resolve MODEL_DIR and ensure tokenizer present in it
#############################################
MODEL_DIR=""
if [[ "${method}" == "ga" ]]; then
  echo "[GA] Merging adapter into base model..."
  [[ -f "${ADAPTER_DIR}/adapter_config.json" ]] || { echo "ERROR: Missing ${ADAPTER_DIR}/adapter_config.json"; exit 1; }
  [[ -f "${BASE_DIR}/config.json" ]] || { echo "ERROR: BASE_DIR is not a full snapshot: ${BASE_DIR}"; exit 1; }

  MERGED_DIR="/mnt/Shared-Storage/sid/checkpoints/merged_wmdp_${unlearning_task}_lat_ga"
  if [[ ! -d "${MERGED_DIR}" ]]; then
    CUDA_VISIBLE_DEVICES=0 python merge_adapter.py --task "${unlearning_task}"
  else
    echo "[GA] Merged model already exists at ${MERGED_DIR}, skipping merge."
  fi
  MODEL_DIR="${MERGED_DIR}"

  # >>> ensure tokenizer files live in the merged folder <<<
  ensure_tokenizer_in_dir "${MODEL_DIR}" "${BASE_DIR}"

elif [[ "${method}" == "rmu" ]]; then
  echo "[RMU] Using provided folder as a full/merged model..."
  [[ -f "${RMU_MODEL_DIR}/config.json" ]] || { echo "ERROR: RMU_MODEL_DIR is not a full model: ${RMU_MODEL_DIR}"; exit 1; }
  MODEL_DIR="${RMU_MODEL_DIR}"

  # >>> ensure tokenizer files live in the RMU folder <<<
  ensure_tokenizer_in_dir "${MODEL_DIR}" "${BASE_DIR}"
else
  echo "ERROR: method must be 'ga' or 'rmu' (got '${method}')" >&2
  exit 1
fi

#############################################
# Build eval args (no need to pass tokenizer overrides)
#############################################
EVAL_ARGS=(
  src/eval.py --config-name=eval.yaml experiment=eval/wmdp/default
  model="${MODEL_NAME}"
  model.model_args.pretrained_model_name_or_path="${MODEL_DIR}"
  +model.model_args.local_files_only=true
  model.model_args.attn_implementation=sdpa
  model.model_args.torch_dtype=bfloat16
  model.model_args.device_map=auto

  # point tokenizer to MODEL_DIR (it now contains tokenizer.json)
  model.tokenizer_args.pretrained_model_name_or_path="${MODEL_DIR}"
  +model.tokenizer_args.local_files_only=true
  +model.tokenizer_args.use_fast=true
  +model.tokenizer_args.legacy=false

  +retain_logs_path="${LOG_DIR}/WMDP_EVAL.json"
  task_name="WMDP_${unlearning_task^^}_LAT_${method^^}"

  # optional: calmer logs with odd RMU keys
  +model.model_args.ignore_mismatched_sizes=true
)

# RMU: force-disable any PEFT/adapter usage (unknown keys are harmless)
if [[ "${method}" == "rmu" ]]; then
  EVAL_ARGS+=(
    '+model.adapter_path=null'
    '+model.adapter_path=""'
    '+model.peft_path=null'
    '+model.use_peft=false'
    '+peft.enabled=false'
  )
fi

echo "running eval.py"
CUDA_VISIBLE_DEVICES=0 python "${EVAL_ARGS[@]}"

conda deactivate
