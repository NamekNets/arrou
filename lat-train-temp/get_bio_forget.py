from datasets import load_dataset

dset = load_dataset("cais/wmdp-bio-forget-corpus", split="train")
dset.to_json("/mnt/Shared-Storage/sid/datasets/wmdp-corpora/bio-forget-corpus.jsonl", force_ascii=False)
