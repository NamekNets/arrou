#!/bin/bash

# Check if domain argument is provided
if [ -z "$1" ]; then
    echo "Usage: bash train_arrou.sh [bio|cyber]"
    echo "Example: bash train_arrou.sh bio"
    exit 1
fi

DOMAIN=$1

# Validate domain argument
if [ "$DOMAIN" != "bio" ] && [ "$DOMAIN" != "cyber" ]; then
    echo "Error: Domain must be 'bio' or 'cyber'"
    echo "Usage: bash train_arrou.sh [bio|cyber]"
    exit 1
fi

# Set paths based on domain
FORGET_CORPUS="/mnt/Shared-Storage/sid/datasets/wmdp-corpora/${DOMAIN}-forget-corpus.jsonl"
OUTPUT_DIR="saves/arrou_model/${DOMAIN}"

echo "Training ARROU model for domain: ${DOMAIN}"
echo "Forget corpus: ${FORGET_CORPUS}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

echo "activating environment"
source /mnt/Shared-Storage/sid/miniconda3/bin/activate
conda activate nameknets

# Set PyTorch CUDA memory allocator to use expandable segments to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "running train_arrou.py"
CUDA_VISIBLE_DEVICES=3 python latent_at/train_arrou.py \
       --model_name meta-llama/Llama-3.2-1B-Instruct \
       --forget_prompts_json ${FORGET_CORPUS} \
       --candidate_layers 4 8 12 16 \
       --refusal_text "I don't know" \
       --output_dir ${OUTPUT_DIR} \
       --use_fp16 \
       --reduce_prefixes 5 \
       --pgd_iters 8 \
       --max_prompts 100

conda deactivate
