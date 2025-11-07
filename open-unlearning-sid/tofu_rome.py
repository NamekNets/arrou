import sys
sys.path.append("./src")
sys.path.append("../EasyEdit")

from src.model import *
from easyeditor import BaseEditor
from easyeditor import ROMEHyperParams

from data.utils import IGNORE_INDEX
from omegaconf import OmegaConf

# Evaluate the model
from src.data import get_datasets, get_collators
# get dictconfig from yaml file


from src.data.utils import preprocess_chat_instance
from transformers import GenerationConfig
import os

# Load composed Hydra config saved by src/train.py (path can be overridden via env)
CFG_PATH = os.environ.get("ROME_CFG", "cfg.json")
config = OmegaConf.load(CFG_PATH)

hparams=ROMEHyperParams.from_hparams('../EasyEdit/hparams/ROME/llama3.2-1B.yaml')
editor=BaseEditor.from_hparams(hparams, cache_dir="/mnt/bigssd/ayan_workspace/pers/proj/models")

# Prepare datasets and collator
datasets = {
    "forget": get_datasets(config.data.forget, template_args=config.model.template_args, tokenizer=editor.tok, predict_with_generate=True),
    "retain": get_datasets(config.data.retain, template_args=config.model.template_args, tokenizer=editor.tok),
}
collators = get_collators(config.collator, tokenizer=editor.tok)

def dataset_item_to_batch(item, tokenizer, model):
    batch = collators([item])
    batch = {k: v.to(model.device) for k, v in batch.items()}
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    input_texts = tokenizer.batch_decode(
        input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    labels = [label[label != IGNORE_INDEX] for label in labels]
    full_texts = tokenizer.batch_decode(
        labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    ground_truths = [
        full_text.replace(input_text, "").strip()
        for input_text, full_text in zip(input_texts, full_texts)
    ]
    return batch, input_texts, ground_truths

def question_str_to_batch(question_str, tokenizer, model):
    item = preprocess_chat_instance(
        tokenizer=tokenizer,
        template_config=config.model.template_args,
        prompt_msgs=[question_str],
        response_msgs=[""],
        max_length=768,
        predict_with_generate=True,
    )
    return dataset_item_to_batch(item, tokenizer, model)

def get_question_answer_from_text(input_text, ground_truth):
    question_str = input_text[:-(len("assistant\n\n"))].split("\n")[-1]
    return question_str, ground_truth


# def get_response_from_base_model(model, tokenizer, o)

if os.environ.get("ROME_DEMO") == "1":
    item = datasets["forget"][0]
    batch, input_texts, ground_truths = dataset_item_to_batch(item, editor.tok, editor.model)
    print(input_texts[0])
    print(ground_truths[0])
    question_str, answer_str = get_question_answer_from_text(input_texts[0], ground_truths[0])
    print("--------------------------------")
    print(question_str)
    print(answer_str)

def generate(batch, model, tokenizer):
    gen_cfg = GenerationConfig(do_sample=True, temperature=0.7, top_p=0.9, max_new_tokens=200, use_cache=True)
    out = model.generate(**batch, generation_config=gen_cfg, pad_token_id=tokenizer.eos_token_id)
    gen_ids = out[:, batch["input_ids"].shape[-1]:]
    return tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

def get_response_from_model(model, tokenizer, question_str):
    item = preprocess_chat_instance(
        tokenizer=tokenizer,
        template_config=config.model.template_args,
        prompt_msgs=[question_str],
        response_msgs=[""],
        max_length=768,
        predict_with_generate=True,
    )
    
    batch, input_texts, ground_truths = dataset_item_to_batch(item, tokenizer, model)
    return generate(batch, model, tokenizer)


FORMAT_SPEC = """\
You are a data normalizer. Convert the given QA into a single JSON object with this schema:

{
  "question": string,        // The original question, trimmed. Do not paraphrase.
  "answer": string,          // The gold answer, as a short string. If it's a list, join with ", ".
  "subject": string          // A short noun phrase naming the entity being asked about.
}

Rules:
- Output ONLY one JSON object. No prose, no markdown, no code fences.
- The subject should be a single word.for example, if the question is "What is the capital of France?", the subject should be "France".
- the "subject" should be in the question exactly. for example, if the question is "Which author lives in Taipei, Taiwan", the subject should be Taipei, Taiwan 
- If the answer is numeric/date/text span, copy it exactly.
"""
def to_prompt(input_text: str) -> str:
    # input_text should include both the original question and the gold answer
    return (
        FORMAT_SPEC
        + "\n\n[BEGIN QA]\n"
        + input_text.strip()
        + "\n[END QA]\n\nJSON:"
    )
    
    
    

import json
import re
from typing import Dict, Tuple

def _first_braced_json(text: str) -> str:
    """
    Extract the first top-level {...} block to survive accidental prose.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found")
    # scan for matching brace
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    raise ValueError("Unbalanced braces in model output")

def _coerce_json(s: str) -> Dict:
    """
    Try strict JSON; if that fails, attempt a few safe normalizations.
    """
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # strip code fences if any
        s2 = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.IGNORECASE)
        try:
            return json.loads(s2)
        except json.JSONDecodeError:
            # normalize single quotes → double quotes cautiously
            s3 = s2
            # only replace quotes around keys and simple string values
            s3 = re.sub(r"(?P<pre>[\{\s,])'(?P<key>[A-Za-z0-9_]+)'\s*:", r'\g<pre>"\g<key>":', s3)
            s3 = re.sub(r':\s*\'(?P<val>[^\'"]*)\'', r': "\g<val>"', s3)
            return json.loads(s3)

def _validate_and_normalize(obj: Dict) -> Tuple[str, str, str]:
    for k in ("question", "answer", "subject"):
        if k not in obj or not isinstance(obj[k], str) or not obj[k].strip():
            raise ValueError(f'Missing or empty "{k}"')
    q = obj["question"].strip()
    a = obj["answer"].strip()
    s = obj["subject"].strip()

    # Normalize whitespace & trivial punctuation
    q = re.sub(r"\s+", " ", q)
    a = re.sub(r"\s+", " ", a)
    s = re.sub(r"\s+", " ", s)

    # Some datasets include trailing periods; keep target clean
    a = a.strip()

    return (q, s, a)  # (question, subject, target)
    
def parse_to_tuple(generated_text: str) -> Tuple[str, str, str]:
    """
    generated_text: raw decode from the model.
    returns: (question, subject, target)
    """
    block = _first_braced_json(generated_text)
    obj = _coerce_json(block)
    q,s, a = _validate_and_normalize(obj)
    # ensure subject is in the question
    if s not in q:
        raise ValueError(f"Subject {s} not in question {q}")
    return q, s, a


def build_edits_from_forget_dataset(forget_dataset):
    prompts = []
    ground_truths = []
    subjects = []

    for i in range(len(forget_dataset)):
        # q = forget_dataset.data[i][question_key]
        # a_val = forget_dataset.data[i][answer_key]
        item = forget_dataset[i]
        batch, input_texts, ground_truths = dataset_item_to_batch(item, editor.tok, editor.model)
        print(input_texts[0])
        print(ground_truths[0])
        question_str, answer_str = get_question_answer_from_text(input_texts[0], ground_truths[0])
        print("--------------------------------")
        print(question_str)
        print(answer_str)
        a_val = question_str + "\n" + answer_str
        if os.environ.get("ROME_DEMO") == "1":
            new_question = to_prompt(question_str + "\n" + answer_str)
        if isinstance(a_val, list):
            a = ", ".join([x.strip() for x in a_val])
        else:
            a = str(a_val).strip()

        prompt = to_prompt(a_val)

        response = None
        tries = 0
        while response is None and tries <3:
            try:
                response = get_response_from_model(editor.model, editor.tok, prompt)
                if response is None:
                    tries += 1
                    continue
                _q, _s, _a = parse_to_tuple(response)
                prompts.append(_q)
                ground_truths.append(_a)
                subjects.append(_s)
            except Exception as e:
                print(f"[warn] Parsing failed for idx={i} (try={tries}): {e}")
                response = None
                tries += 1

        if response is None:
            print(f"[skip] Could not normalize QA at idx={i}; skipping.")
            continue

    return prompts, ground_truths, subjects


def main():
    question_key = getattr(config, "question_key", "question")
    answer_key = "answer"

    forget_dataset = datasets["forget"]

    print(f"Preparing ROME edits for {len(forget_dataset)} forget items...")
    prompts, gts, subs = build_edits_from_forget_dataset(
        forget_dataset
    )

    if len(prompts) == 0:
        raise RuntimeError("No valid edits prepared from forget dataset.")

    target_new = ["cant say" for _ in prompts]

    print(f"Running ROME sequential edits over {len(prompts)} items...")
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=gts,
        target_new=target_new,
        subject=subs,
        sequential_edit=True,
    )

    task_name = getattr(config, "task_name", None)
    if task_name is None:
        forget_split = getattr(config, "forget_split", "forget")
        task_name = f"tofu_ROME_{forget_split}"

    out_dir = os.path.join("saves", "unlearn", task_name)
    os.makedirs(out_dir, exist_ok=True)

    try:
        edited_model.save_pretrained(out_dir)
    except Exception as e:
        print(f"[warn] save_pretrained failed on edited_model: {e}. Trying base editor model.")
        editor.model.save_pretrained(out_dir)
    try:
        editor.tok.save_pretrained(out_dir)
    except Exception as e:
        print(f"[warn] Failed to save tokenizer: {e}")

    with open(os.path.join(out_dir, "rome_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved edited model to: {out_dir}")
    print("Run eval via src/eval.py with model.model_args.pretrained_model_name_or_path=", out_dir)


if __name__ == "__main__":
    main()

# item = datasets["forget"][1]
# batch, input_texts, ground_truths = dataset_item_to_batch(item, editor.tok, editor.model)
# print(input_texts[0])
# print(ground_truths[0])
# question_str, answer_str = get_question_answer_from_text(input_texts[0], ground_truths[0])
# print("--------------------------------")
# print(question_str)
# print(answer_str)


# if os.environ.get("ROME_DEMO") == "1":
#     new_question = to_prompt(question_str + "\n" + answer_str)
#     response = None
#     tries = 0
#     while response is None and tries < 5:
#         try:
#             print(f"Trying {tries} times")
#             response = get_response_from_model(editor.model, editor.tok, new_question)
#             if response is None:
#                 print("No response, retrying...")
#                 tries += 1
#                 continue
#             print(response)
#             question, subject, target = parse_to_tuple(response)
#             print(question, subject, target)
#         except Exception as e:
#             print(f"Error on try {tries}: {e}")
#             response = None
#             tries += 1
#             continue

#     prompts = [question]
#     ground_truths = [target]
#     subjects = [subject]
#     new_targets = ["cant say"]
#     print(get_response_from_model(editor.model, editor.tok, question_str))
#     metrics, edited_model, _ = editor.edit(
#         prompts=prompts,
#         ground_truth=ground_truths,
#         target_new=new_targets,
#         subject=subjects,
#         sequential_edit=True,
#     )
#     print(get_response_from_model(edited_model, editor.tok, question_str))