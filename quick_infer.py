import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# --------- EDIT THESE ---------
MODEL_DIR = "./mrome_lat_model" #"/scratch2/sky/hf_cache/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6" # "./mrome_lat_model" # path to your edited model dir
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"  # only used if MODEL_DIR lacks a valid config
PROMPT = "Who invented the character of Hermione Granger?"#"Japan is also known as the land of the rising sun. What is the capital of the country that attacked Pearl Harbor?"#"Who invented the character of Hermione Granger in the Harry Potter series?"
MAX_NEW_TOKENS = 128
DO_SAMPLE = False  # set True if you want sampling
# --------------------------------


def load_model_and_tokenizer(model_dir: str, base_model: str | None = None):
    cache_dir = os.environ.get("HF_HOME", None)
    config_path = os.path.join(model_dir, "config.json")

    cfg_valid = False
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            cfg_valid = isinstance(cfg, dict) and ("model_type" in cfg or "architectures" in cfg)
        except Exception:
            cfg_valid = False

    if cfg_valid:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, device_map="auto", trust_remote_code=True, cache_dir=cache_dir
        )
        tok = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True, cache_dir=cache_dir
        )
    else:
        if base_model is None:
            raise ValueError("MODEL_DIR lacks a valid config.json. Set BASE_MODEL to a repo id to load weights.")
        from safetensors.torch import load_file as load_safetensors

        weights_path = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"No model.safetensors in {model_dir}")
        state_dict = load_safetensors(weights_path)
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map="auto", trust_remote_code=True, cache_dir=cache_dir, state_dict=state_dict
        )
        tok_source = model_dir if os.path.exists(os.path.join(model_dir, "tokenizer.json")) else base_model
        tok = AutoTokenizer.from_pretrained(
            tok_source, trust_remote_code=True, cache_dir=cache_dir
        )

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, tok


def format_inputs(tok: AutoTokenizer, prompt: str):
    if hasattr(tok, "apply_chat_template"):
        chat = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        return tok(chat, return_tensors="pt")
    return tok(prompt, return_tensors="pt", add_special_tokens=True)


@torch.inference_mode()
def generate(model, tok, prompt: str, max_new_tokens: int = 128, do_sample: bool = False):
    enc = format_inputs(tok, prompt)
    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_p=1.0,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    input_len = enc["input_ids"].shape[1]
    return tok.decode(gen[0][input_len:], skip_special_tokens=True)


def main():
    # Prefer HF_HOME if set; TRANSFORMERS_CACHE is deprecated in v5
    os.environ.setdefault("HF_HOME", "/scratch2/sky/hf_cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])  # still honored pre-v5
    os.environ.setdefault("HF_HUB_CACHE", os.environ["HF_HOME"]) 

    model, tok = load_model_and_tokenizer(MODEL_DIR, BASE_MODEL)
    out = generate(model, tok, PROMPT, MAX_NEW_TOKENS, DO_SAMPLE)
    print("Prompt:", PROMPT)
    print("Output:\n" + out.strip())


if __name__ == "__main__":
    main()


