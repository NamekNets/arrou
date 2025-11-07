from typing import List, Optional, Union

import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


def load_targeted_wmdp_data(
        retain_corpora: List[str]=['bio-forget-corpus', 'cyber-forget-corpus'],
        forget_corpora: List[str]=['wikitext', 'wikitext'])-> list[Dataset, Dataset]:
    """
    Forget and retain datasets differ. Load forget and retain then recombine into columns
    'adv_tokens' and 'def_tokens' in new Huggingface Dataset object.

    Supports bio/cyber WMDP retain corpora and WikiText (paired with both bio and cyber unlearn)
    as options for retain dataset. See config line 'retain_corpora'.
    """

    # Load and rename datasets for 'forget' corpora
    # Only bio forget needs to be locally loaded
    hf_location = "cais/wmdp-corpora"
    # hf_location = "/mnt/Shared-Storage/sid/datasets/wmdp-corpora"
    forget_data, retain_data, all_data = [], [], []

    for d in forget_corpora:
        if d == "bio-forget-corpus": # wmdp bio must be downloaded
            dataset_path = f"/mnt/Shared-Storage/sid/datasets/wmdp-corpora/{d}.jsonl"
            forget_dataset = load_dataset('json', data_files=dataset_path, split='train')
            forget_dataset = forget_dataset.rename_column('text', 'adv_tokens')
            forget_data.append(forget_dataset)
        elif d == "cyber-forget-corpus":
            forget_dataset = load_dataset(hf_location, name=d, split='train')
            forget_dataset = forget_dataset.rename_column('text', 'adv_tokens')
        else:
            raise NotImplementedError
        forget_data.append(forget_dataset)

    for d in retain_corpora:
        if d in ['bio-retain-corpus', 'cyber-retain-corpus']:
            retain_dataset = load_dataset(hf_location, name=d, split='train')
            retain_dataset = retain_dataset.rename_column('text', 'def_tokens')
        elif d in ['wikitext']:
            retain_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            retain_dataset = retain_dataset.rename_column('text', 'def_tokens')
        else:
            raise NotImplementedError
        retain_data.append(retain_dataset)

    def merge_rows(example1, example2):
        return {'adv_tokens': example1['adv_tokens'], 'def_tokens': example2['def_tokens']}

    for fd, rd in zip(forget_data, retain_data):
        min_length = min(len(fd), len(rd))
        dset = fd.select(range(min_length)).map(
            lambda x,
                   idx: merge_rows(x, rd[idx]),
            with_indices=True,
        )
        if 'title' in dset:  # if bio
            dset = dset.remove_columns(['title', 'abstract', 'doi'])
        all_data.append(dset)

    return all_data


def make_targeted_wmdp_dataloader(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    lat_batch_size: int=4,
    data_truncate_length: int=600,
) -> tuple[DataLoader, DataLoader]:

    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=lat_batch_size,
        collate_fn=WMDPLATTargetedDataCollator(tokenizer, truncate_length=data_truncate_length),
        drop_last=True,
    )
    return dataloader


def make_untargeted_wmdp_dataloaders(
        data: Union[list[str], list[list[str]]],
        tokenizer: AutoTokenizer,
        sft: Optional[bool] = True,
        sft_batch_size: int=4,
        lat_batch_size: int=4,
        data_truncate_length: int=600,
) -> Union[DataLoader, tuple[DataLoader, DataLoader]]:
    """
    Don't use for anything except WMDP unlearn corpora exactly as loaded with tuple list [bio, cyber].
    Used for two things: creating SFT dataloaders, and creating WMDP dataloaders in the case where defence and attacker both train on the WMDP unlearn corpora.

    In the paper, by default, we use targeted, where the defence trains toward WikiText and the attack trains toward WMDP unlearn corpora. The away losses are gradient ascent on these same datasets but swapped between attack and defence.

    Args:
        config: OmegaConf object created from yaml file.
        data: Each list this contains will be one of the bio/cyber datasets.
        sft: If True, data list is two copies of the same dataset. This will only be used to generate supervised fine tuning dataloader for SFT portion of R2D2 loss in LAT.
    Returns:
        Dataloaders.
    """
    if sft:
        return DataLoader(
            data,
            shuffle=True,
            batch_size=sft_batch_size,
            collate_fn=WMDPLATDataCollator(tokenizer, truncate_length=data_truncate_length),
            drop_last=True,
        )

    wmdp_bio_dataloader = DataLoader(
        data[0],
        shuffle=True,
        batch_size=lat_batch_size,
        collate_fn=WMDPLATDataCollator(tokenizer, truncate_length=data_truncate_length),
        drop_last=True,
    )
    wmdp_cyber_dataloader = DataLoader(
        data[1],
        shuffle=True,
        batch_size=lat_batch_size,
        collate_fn=WMDPLATDataCollator(tokenizer, truncate_length=data_truncate_length),
        drop_last=True,
    )
    return wmdp_bio_dataloader, wmdp_cyber_dataloader


def load_sft_dataset(sft_corpora: str='alpaca') -> list[str]:
    """Works for wikitext and alpaca."""
    if sft_corpora == "wikitext":
        raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    elif sft_corpora == "alpaca":
        raw_data = load_dataset("tatsu-lab/alpaca", "default", split="train")
    else:
        raise NotImplementedError
    data = []
    for x in raw_data:
        data.append(str(x['text']))
    return data


class WMDPLATTargetedDataCollator:
    """
    Targeted version of below class, which returns *different* adv_labels and def_labels
    using wmdp retain and unlearn corpora.
    Specifically designed to WMDP corpora data, working with data loading methods from jsonl in wmdp/cut/utils.py,
    with batching removed as batching is done here instead.
    This class is not used for SFT.
    """

    def __init__(self, tokenizer, truncate_length: Optional[int] = None):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.eos_token_id
        self.truncate_length = truncate_length

    def __call__(self, batch: List[str]):
        B = len(batch)
        tokenized_def_inputs = [self.tokenizer(example["def_tokens"])["input_ids"] for example in batch]
        tokenized_adv_inputs = [self.tokenizer(example["adv_tokens"])["input_ids"] for example in batch]
        def_lengths = [len(x) for x in tokenized_def_inputs]
        adv_lengths = [len(x) for x in tokenized_adv_inputs]
        pad_length = max(max(adv_lengths), max(def_lengths))

        def_tokens = torch.zeros(B, pad_length, dtype=torch.long)
        adv_tokens = torch.zeros(B, pad_length, dtype=torch.long)
        def_labels_mask = torch.zeros(B, pad_length, dtype=torch.bool)
        adv_labels_mask = torch.zeros(B, pad_length, dtype=torch.bool)

        for i, (def_inputs, adv_inputs) in enumerate(zip(tokenized_def_inputs, tokenized_adv_inputs)):
            def_tokens[i] = torch.tensor(def_inputs + [self.pad_token_id] * (pad_length - def_lengths[i]),
                                         dtype=torch.long)
            adv_tokens[i] = torch.tensor(adv_inputs + [self.pad_token_id] * (pad_length - adv_lengths[i]),
                                         dtype=torch.long)
            def_labels_mask[i, :def_lengths[i]] = True
            adv_labels_mask[i, :adv_lengths[i]] = True

        if self.truncate_length is not None:
            def_tokens = def_tokens[:, :self.truncate_length]
            adv_tokens = adv_tokens[:, :self.truncate_length]
            def_labels_mask = def_labels_mask[:, :self.truncate_length]
            adv_labels_mask = adv_labels_mask[:, :self.truncate_length]

        return {
            "def_tokens": def_tokens,
            "adv_tokens": adv_tokens,
            "def_labels_mask": def_labels_mask,
            "adv_labels_mask": adv_labels_mask,
        }


class WMDPLATDataCollator:
    """
    Specifically designed to WMDP corpora data, working with data loading methods from jsonl in wmdp/cut/utils.py,
    with batching removed as batching is done here instead.

    Note adv_labels == def_labels because we just do a 1-p loss for the defence on the 'bad corpora'.
    This class is used for both SFT and WMDP unlearn corpora.
    For SFT, it suffices to have the labels mask be created as per usual.
    """

    def __init__(self, tokenizer, truncate_length):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.eos_token_id
        self.truncate_length = truncate_length

    def __call__(self, batch: List[str]):
        B = len(batch)
        tokenized_inputs = [self.tokenizer(example)["input_ids"] for example in batch]
        lengths = [len(example) for example in tokenized_inputs]
        pad_length = max(lengths)

        tokens = torch.zeros(B, pad_length, dtype=torch.long)
        adv_labels_mask = torch.zeros(B, pad_length, dtype=torch.bool)
        def_labels_mask = torch.zeros(B, pad_length, dtype=torch.bool)

        for i, example in enumerate(tokenized_inputs):
            l = lengths[i]
            tokens[i] = torch.tensor(example + [self.pad_token_id] * (pad_length - l), dtype=torch.long)
            adv_labels_mask[i, :l] = True
            def_labels_mask[i, :l] = True

        if self.truncate_length is not None:
            tokens = tokens[:, :self.truncate_length]
            def_labels_mask = def_labels_mask[:, :self.truncate_length]
            adv_labels_mask = adv_labels_mask[:, :self.truncate_length]

        return {
            "tokens": tokens,
            "def_labels_mask": def_labels_mask,
            "adv_labels_mask": adv_labels_mask,
        }
