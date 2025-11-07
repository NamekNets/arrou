# CUDA_VISIBLE_DEVICES=0 python tofu_experiment.py \
#   --model open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
#   --forget_jsonl /path/to/tofu_forget.jsonl \
#   --retain_jsonl /path/to/tofu_retain.jsonl \
#   --limit 50 --rounds 3


# CUDA_VISIBLE_DEVICES=0 python tofu_experiment.py \
#   --model open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
#   --forget_jsonl /scratch2/sky/ml/anlp/cp_fin/data/tofu/324592d84ae4f482ac7249b9285c2ecdb53e3a68 \
#   --limit 50 --rounds 3

CUDA_VISIBLE_DEVICES=0 python tofu_experiment.py \
  --model open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
  --forget_jsonl /scratch2/sky/ml/anlp/cp_fin/data/tofu/324592d84ae4f482ac7249b9285c2ecdb53e3a68 \
  --limit 40 --rounds 3 \
  --normalize_subjects \
  --subject_model meta-llama/Llama-3.2-1B-Instruct