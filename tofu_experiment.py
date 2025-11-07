import os
import json
import argparse
from typing import List, Tuple, Iterable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mrome_lat_integration import minimax_loop


def read_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_tofu_jsonl(
    forget_path: str,
    retain_path: Optional[str] = None,
    limit: Optional[int] = None,
) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    forget: List[Tuple[str, str, str]] = []
    retain: List[str] = []

    # Expect fields: question, subject, answer (common in TOFU dumps)
    for ex in read_jsonl(forget_path):
        q = ex.get("question") or ex.get("prompt") or ex.get("input")
        subj = ex.get("subject") or ex.get("entity") or ""
        ans = ex.get("answer") or ex.get("target") or ex.get("label")
        if not (q and ans):
            continue
        forget.append((q, subj, ans))
        if limit is not None and len(forget) >= limit:
            break

    if retain_path is not None and os.path.exists(retain_path):
        for ex in read_jsonl(retain_path):
            rp = ex.get("question") or ex.get("prompt") or ex.get("input")
            if rp:
                retain.append(rp)
    else:
        # Fallback retain prompts (simple GK)
        retain = [
            "What is the capital of France?",
            "What is 2 + 2?",
            "Who painted the Mona Lisa?",
            "Name the largest planet in our solar system.",
            "What language is primarily spoken in Spain?",
        ]

    return forget, retain


FORMAT_SPEC = (
    "You are a data normalizer. Convert the given QA into a single JSON object with this schema:\n\n"
    "{\n  \"question\": string,\n  \"answer\": string,\n  \"subject\": string\n}\n\n"
    "Rules:\n"
    "- Output ONLY one JSON object. No prose, no markdown, no code fences.\n"
    "- The subject should be a single word.\n"
    "- The subject is not the one that is used in grammatical sense, it is the one that which the answer depends on. It is supposed to be the entity that is being asked about.\n"
    "- The subject must appear verbatim in the question.\n"
    "- If the answer is numeric/date/text span, copy it exactly.\n"
)


@torch.inference_mode()
def build_subject_model(repo_id: str, cache_dir: Optional[str] = None):
    model = AutoModelForCausalLM.from_pretrained(
        repo_id, device_map="auto", trust_remote_code=True, cache_dir=cache_dir, attn_implementation="eager"
    )
    tok = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True, cache_dir=cache_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, tok


def extract_subject(model, tok, question: str, answer: str) -> str:
    prompt = (
        f"{FORMAT_SPEC}\n"
        f"Q: {question}\n"
        f"A: {answer}\n"
    )
    if hasattr(tok, "apply_chat_template"):
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        enc = tok(text, return_tensors="pt")
    else:
        enc = tok(prompt, return_tensors="pt")
    # Ensure inputs on same device as model's first parameter
    try:
        dev = next(model.parameters()).device
        enc = {k: v.to(dev) for k, v in enc.items()}
    except StopIteration:
        pass
    gen = model.generate(
        **enc, max_new_tokens=128, do_sample=False, top_p=1.0, eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id
    )
    input_len = enc["input_ids"].shape[1]
    out = tok.decode(gen[0][input_len:], skip_special_tokens=True).strip()
    # Robust JSON slice
    try:
        start = out.find("{")
        end = out.rfind("}")
        if start != -1 and end != -1:
            out = out[start : end + 1]
        obj = json.loads(out)
        subj = str(obj.get("subject", "")).strip()
    except Exception:
        subj = ""
    # Enforce constraints
    if subj and (subj in question):
        # Single word heuristic: pick the longest token that appears
        subj = subj.split()[0] if " " in subj else subj
        return subj
    # fallback: pick last capitalized token in question
    toks = [t.strip("?,.!\"'()") for t in question.split()]
    for t in reversed(toks):
        if t and t[0].isupper():
            return t
    return ""


def pick_default_files(root: str) -> Tuple[str, Optional[str]]:
    # Prefer specific splits if present
    candidates_forget = [
        "forget01.json",
        "forget05.json",
        "forget10.json",
        "full.json",
    ]
    candidates_retain = [
        "retain99.json",
        "retain95.json",
        "retain90.json",
    ]
    f_path = None
    r_path = None
    for nm in candidates_forget:
        p = os.path.join(root, nm)
        if os.path.exists(p):
            f_path = p
            break
    for nm in candidates_retain:
        p = os.path.join(root, nm)
        if os.path.exists(p):
            r_path = p
            break
    if f_path is None:
        raise FileNotFoundError(f"No default forget file found in {root}")
    return f_path, r_path


def main():
    parser = argparse.ArgumentParser(description="Run mROMΕ on TOFU-style data and report metrics")
    parser.add_argument("--model", type=str, required=True, help="Base model repo id (e.g., meta-llama/Llama-3.2-1B-Instruct)")
    parser.add_argument("--forget_jsonl", type=str, required=True, help="Path to TOFU forget JSONL OR a directory containing TOFU JSON files")
    parser.add_argument("--retain_jsonl", type=str, default=None, help="Optional retain JSONL; if --forget_jsonl is a directory and this is omitted, a default retain split will be chosen if present")
    parser.add_argument("--limit", type=int, default=10, help="Max forget examples to use")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--normalize_subjects", action="store_true", help="Use a clean Llama to generate subject strings")
    parser.add_argument("--subject_model", type=str, default=None, help="Repo id for subject extraction (defaults to --model)")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", "/scratch2/sky/hf_cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])  # honored pre-v5
    os.environ.setdefault("HF_HUB_CACHE", os.environ["HF_HOME"]) 

    forget_path = args.forget_jsonl
    retain_path = args.retain_jsonl
    if os.path.isdir(forget_path):
        forget_path, auto_retain = pick_default_files(forget_path)
        if retain_path is None:
            retain_path = auto_retain
            if retain_path is None:
                print("No retain file found in directory; will use fallback retain prompts.")
        print({"forget": forget_path, "retain": retain_path})

    forget_set, retain_set = load_tofu_jsonl(forget_path, retain_path, args.limit)

    # Optionally generate subjects using a clean copy of the base model
    if args.normalize_subjects:
        subj_repo = args.subject_model or args.model
        subj_model, subj_tok = build_subject_model(subj_repo, os.environ.get("HF_HOME"))
        new_forget = []
        for q, s, a in forget_set:

            if s:
                new_forget.append((q, s, a))
                continue
            s_gen = extract_subject(subj_model, subj_tok, q, a)
            print(f"Question: {q}, Subject: {s_gen}, Answer: {a}")
            new_forget.append((q, s_gen, a))
        forget_set = new_forget
    print(f"Loaded forget={len(forget_set)} retain={len(retain_set)}")

    # Run minimax loop; script will save weights and per-round generations under ./mrome_lat_model
    minimax_loop(
        model_name=args.model,
        forget_dataset=forget_set,
        retain_dataset=retain_set,
        rounds=args.rounds,
    )

    # Post-hoc: compute a simple refusal rate on forget prompts from saved outputs of last round
    out_dir = "./mrome_lat_model"
    last_path = os.path.join(out_dir, f"outputs_round{args.rounds}.json")
    refuse_text = "I'm sorry, but I can't answer that."
    if os.path.exists(last_path):
        outs = json.load(open(last_path))
        total = len(outs)
        refuse = sum(1 for o in outs if refuse_text.lower() in o.get("generation", "").lower())
        print({"round": args.rounds, "refusal_rate": (refuse / max(total, 1)), "n": total, "path": last_path})
    else:
        print(f"No outputs found at {last_path}; skipping refusal metric.")

    # Re-generate clean outputs with chat template to avoid any decoding artifacts
    try:
        edited_model = AutoModelForCausalLM.from_pretrained(out_dir, device_map="auto", trust_remote_code=True)
        edited_tok = AutoTokenizer.from_pretrained(out_dir, trust_remote_code=True)
        if edited_tok.pad_token_id is None:
            edited_tok.pad_token = edited_tok.eos_token
        gens = []
        dev = next(edited_model.parameters()).device
        with torch.inference_mode():
            for q, _, _ in forget_set:
                if hasattr(edited_tok, "apply_chat_template"):
                    chat = edited_tok.apply_chat_template(
                        [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
                    )
                    enc = edited_tok(chat, return_tensors="pt")
                else:
                    enc = edited_tok(q, return_tensors="pt", add_special_tokens=True)
                enc = {k: v.to(dev) for k, v in enc.items()}
                gen = edited_model.generate(
                    **enc,
                    max_new_tokens=128,
                    do_sample=False,
                    top_p=1.0,
                    eos_token_id=edited_tok.eos_token_id,
                    pad_token_id=edited_tok.eos_token_id,
                )
                input_len = enc["input_ids"].shape[1]
                text = edited_tok.decode(gen[0][input_len:], skip_special_tokens=True)
                gens.append({"prompt": q, "generation": text})
        with open(os.path.join(out_dir, f"outputs_round{args.rounds}_clean.json"), "w") as f:
            json.dump(gens, f, indent=2)
        print({"clean_outputs": os.path.join(out_dir, f"outputs_round{args.rounds}_clean.json")})
    except Exception as e:
        print(f"Clean regeneration skipped due to error: {e}")

    # ---- Full metrics: Forget QA prob, Forget ROUGE, retain utility/ROUGE ----
    def lcs_len(a_tokens, b_tokens):
        m, n = len(a_tokens), len(b_tokens)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            ai = a_tokens[i-1]
            for j in range(1, n+1):
                if ai == b_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = dp[i-1][j] if dp[i-1][j] >= dp[i][j-1] else dp[i][j-1]
        return dp[m][n]

    def rouge_l_f1(ref: str, hyp: str) -> float:
        ref_t = ref.strip().split()
        hyp_t = hyp.strip().split()
        if not ref_t or not hyp_t:
            return 0.0
        lcs = lcs_len(ref_t, hyp_t)
        prec = lcs / len(hyp_t)
        rec = lcs / len(ref_t)
        if prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)

    @torch.inference_mode()
    def answer_geomean_prob(model, tok, prompt: str, answer: str, device: torch.device) -> float:
        full = tok(prompt + answer, return_tensors="pt").to(device)
        pr = tok(prompt, return_tensors="pt").to(device)
        labels = full["input_ids"].clone()
        labels[:, : pr["input_ids"].shape[1]] = -100
        out = model(**full, labels=labels)
        loss = float(out.loss.detach().cpu())
        # geometric mean probability per token
        try:
            import math
            return math.exp(-loss)
        except Exception:
            return 0.0

    metrics_path = os.path.join(out_dir, f"metrics_round{args.rounds}.json")
    try:
        # Load edited model for scoring
        edited_model = AutoModelForCausalLM.from_pretrained(out_dir, device_map="auto", trust_remote_code=True)
        edited_tok = AutoTokenizer.from_pretrained(out_dir, trust_remote_code=True)
        if edited_tok.pad_token_id is None:
            edited_tok.pad_token = edited_tok.eos_token
        dev = next(edited_model.parameters()).device

        # Forget metrics
        forget_probs = []
        forget_rouges = []
        forget_refusal = 0
        for q, s, a in forget_set:
            # Generate
            if hasattr(edited_tok, "apply_chat_template"):
                chat = edited_tok.apply_chat_template(
                    [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
                )
                enc = edited_tok(chat, return_tensors="pt")
            else:
                enc = edited_tok(q, return_tensors="pt", add_special_tokens=True)
            enc = {k: v.to(dev) for k, v in enc.items()}
            gen = edited_model.generate(
                **enc,
                max_new_tokens=128,
                do_sample=False,
                top_p=1.0,
                eos_token_id=edited_tok.eos_token_id,
                pad_token_id=edited_tok.eos_token_id,
            )
            input_len = enc["input_ids"].shape[1]
            hyp = edited_tok.decode(gen[0][input_len:], skip_special_tokens=True)
            if refuse_text.lower() in hyp.lower():
                forget_refusal += 1
            forget_rouges.append(rouge_l_f1(a, hyp))
            forget_probs.append(answer_geomean_prob(edited_model, edited_tok, q, a, dev))

        forget_metrics = {
            "n": len(forget_set),
            "refusal_rate": (forget_refusal / max(1, len(forget_set))),
            "avg_answer_prob": sum(forget_probs)/max(1, len(forget_probs)),
            "avg_rougeL": sum(forget_rouges)/max(1, len(forget_rouges)),
        }

        # Retain metrics (if retain file has answers)
        retain_metrics = {"n": len(retain_set), "avg_rougeL": None}
        if isinstance(retain_path, str) and retain_path and os.path.exists(retain_path):
            try:
                retain_pairs = []
                for ex in read_jsonl(retain_path):
                    q = ex.get("question") or ex.get("prompt") or ex.get("input")
                    a = ex.get("answer") or ex.get("target") or ex.get("label")
                    if q and a:
                        retain_pairs.append((q, a))
                if retain_pairs:
                    r_rouges = []
                    for q, a in retain_pairs:
                        if hasattr(edited_tok, "apply_chat_template"):
                            chat = edited_tok.apply_chat_template(
                                [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
                            )
                            enc = edited_tok(chat, return_tensors="pt")
                        else:
                            enc = edited_tok(q, return_tensors="pt", add_special_tokens=True)
                        enc = {k: v.to(dev) for k, v in enc.items()}
                        gen = edited_model.generate(
                            **enc,
                            max_new_tokens=128,
                            do_sample=False,
                            top_p=1.0,
                            eos_token_id=edited_tok.eos_token_id,
                            pad_token_id=edited_tok.eos_token_id,
                        )
                        input_len = enc["input_ids"].shape[1]
                        hyp = edited_tok.decode(gen[0][input_len:], skip_special_tokens=True)
                        r_rouges.append(rouge_l_f1(a, hyp))
                    retain_metrics["avg_rougeL"] = sum(r_rouges)/max(1, len(r_rouges))
            except Exception as e:
                print(f"Retain metrics skipped: {e}")

        all_metrics = {"round": args.rounds, "forget": forget_metrics, "retain": retain_metrics}
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print({"metrics": metrics_path})
    except Exception as e:
        print(f"Metrics computation skipped due to error: {e}")


if __name__ == "__main__":
    main()


