import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, os

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True, choices=["bio", "cyber"], help="Task name")
args = parser.parse_args()

print(args)

base_dir = "/mnt/Shared-Storage/HF_CACHE/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6"
adapter_dir = f"/home/sid/nameknets/unlearning_wmdp_{args.task}_ga_test_save"
out_dir = f"/mnt/Shared-Storage/sid/checkpoints/merged_wmdp_{args.task}_lat_ga"

tok = AutoTokenizer.from_pretrained(base_dir, use_fast=False, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    base_dir, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True
)
model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
model = model.merge_and_unload()  # folds LoRA into base weights

os.makedirs(out_dir, exist_ok=True)
model.save_pretrained(out_dir)
tok.save_pretrained(out_dir)
print("Saved merged model to", out_dir)
