import pickle
import torch
import os
from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule
from transformers import DataCollatorForLanguageModeling
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizerFast
from hydra.utils import get_original_cwd, to_absolute_path
from typing import Optional, Union
from litgpt.tokenizer import Tokenizer
from tqdm import tqdm
import os
from datasets import Dataset, DatasetDict
import numpy as np
from collections import defaultdict

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Datamodule(LightningDataModule):
    def __init__(self, dataset, batch_size, num_workers, tokenizer, small_val):
        super(Datamodule, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.small_val = small_val
        self.collate_fn_pad = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    def setup(self, stage=None):
        self.train_dataset = self.dataset["train"]
        self.val_dataset = self.dataset["val"]
        self.test_dataset = self.dataset["test"]

    def collate_fn_pad(self, batch):
        x, y = zip(*batch)
        # Pad sequences to the maximum length in the batch
        x_padded = pad_sequence(x, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)
        return x_padded, y_padded

    def connect(self, max_seq_length: Optional[int] = None) -> None:
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length
        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.collate_fn_pad,
        )

    def val_dataloader(self):
        val1 = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.collate_fn_pad,
        )
        val2 = DataLoader(
            self.small_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.collate_fn_pad,
        )

        return [val1, val2]

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.collate_fn_pad,
        )


def get_data(cfg: DictConfig, tokenizer, num_bins=10):
    print("Loading dataset...")
    train_file = to_absolute_path(os.path.join(cfg.data.datapath, cfg.data.train_file))
    val_file = to_absolute_path(os.path.join(cfg.data.datapath, cfg.data.val_file))

    hf_dataset = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "val": val_file,
            "test": val_file,
        },
    )
    print(
        f"Dataset loaded. Sizes - Train: {len(hf_dataset['train'])}, Val: {len(hf_dataset['val'])}, Test: {len(hf_dataset['test'])}"
    )

    # Calculate bin ranges
    bin_size = 4096 // num_bins
    bins = [(i * bin_size, (i + 1) * bin_size - 1) for i in range(num_bins)]
    print(f"Created {num_bins} bins: {bins}")

    # Initialize bin assignments
    bin_assignments = defaultdict(lambda: defaultdict(list))
    samples_needed = {
        "train": int(cfg.data["num_train"]) // num_bins,
        "val": int(cfg.data["num_val"]),
    }
    test_samples = int(cfg.eval["num_examples"])  # Full test set size
    print(
        f"Samples needed per bin: {samples_needed}, test samples total: {test_samples}"
    )

    def get_bin_idx(seq_len):
        for idx, (start, end) in enumerate(bins):
            if start <= seq_len <= end:
                return idx
        return num_bins - 1

    def bin_needs_split(bin_idx, split):
        return len(bin_assignments[bin_idx][split]) < samples_needed[split]

    def split_needs_samples(split):
        return any(bin_needs_split(bin_idx, split) for bin_idx in range(num_bins))

    # First process test set - will be shared across all bins
    print("\nProcessing test split...")
    test_examples = []
    for example in hf_dataset["test"]:
        if len(test_examples) >= test_samples:
            break

        text = (
            tokenizer.bos_token + example["search_path"].strip() + tokenizer.eos_token
        )
        outputs = tokenizer(
            text,
            truncation=True,
            max_length=cfg.model.block_size,
            padding="max_length",
            return_overflowing_tokens=False,
        )
        test_examples.append(outputs["input_ids"])

        if len(test_examples) % 1000 == 0:
            print(f"Processed {len(test_examples)} test examples")

    # Assign same test examples to all bins
    for bin_idx in range(num_bins):
        bin_assignments[bin_idx]["test"] = test_examples

    # Process train and val splits
    for split in ["train", "val"]:
        print(f"\nProcessing {split} split...")
        examples_processed = 0

        for example in hf_dataset[split]:
            if not split_needs_samples(split):
                print(f"All bins full for {split}, moving to next split")
                break

            text = (
                tokenizer.bos_token
                + example["search_path"].strip()
                + tokenizer.eos_token
            )
            outputs = tokenizer(
                text,
                truncation=True,
                max_length=cfg.model.block_size,
                padding="max_length",
                return_overflowing_tokens=False,
            )

            try:
                seq_len = outputs["input_ids"].index(tokenizer.eos_token_id) + 1
            except ValueError:
                seq_len = len(outputs["input_ids"])
            seq_len = min(seq_len, 4096)

            bin_idx = get_bin_idx(seq_len)
            if bin_needs_split(bin_idx, split):
                bin_assignments[bin_idx][split].append(outputs["input_ids"])

            examples_processed += 1
            if examples_processed % 1000 == 0:
                print(f"Processed {examples_processed} examples")
                print("\nCurrent bin states:")
                for b_idx in range(num_bins):
                    print(
                        f"Bin {b_idx}: {len(bin_assignments[b_idx][split])}/{samples_needed[split]}"
                    )

    # Create curriculum datasets
    curriculum_datasets = []
    for bin_idx in range(num_bins):
        bin_datasets = {}
        for split in ["train", "val", "test"]:
            bin_datasets[split] = Dataset.from_dict(
                {"input_ids": bin_assignments[bin_idx][split]}
            )
        curriculum_datasets.append(DatasetDict(bin_datasets))

    print("\nFinal bin statistics:")
    for bin_idx, (start, end) in enumerate(bins):
        print(f"\nBin {bin_idx} ({start}-{end}):")
        for split in ["train", "val"]:
            print(
                f"  {split}: {len(bin_assignments[bin_idx][split])}/{samples_needed[split]}"
            )
        print(f"  test: {len(bin_assignments[bin_idx]['test'])}")

    return curriculum_datasets


def get_tokenizer(tok_data: DictConfig):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=to_absolute_path(tok_data.tokenizer_path)
    )
    tokenizer.eos_token = "[EOS]"
    tokenizer.unk_token = "[UNK]"
    tokenizer.pad_token = "[PAD]"
    tokenizer.mask_token = "[MASK]"
    tokenizer.bos_token = "[BOS]"
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
