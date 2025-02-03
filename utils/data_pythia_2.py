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
from collections import defaultdict
import os
from datasets import Dataset, DatasetDict

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Datamodule(LightningDataModule):
    def __init__(self, dataset, batch_size, num_workers, tokenizer):
        super(Datamodule, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn_pad = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    def collate_fn_pad(self, batch):
        x, y = zip(*batch)
        # Pad sequences to the maximum length in the batch
        x_padded = pad_sequence(x, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)
        return x_padded, y_padded

    def connect(self, max_seq_length: Optional[int] = None) -> None:
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length
        return self

    def setup(self, stage=None):
        self.train_dataset = self.dataset["train"]
        self.val_dataset = self.dataset["val"]
        self.test_first_dataset = self.dataset["test_first"]
        self.test_last_dataset = self.dataset["test_last"]

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
        # Return multiple validation dataloaders
        return [
            DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=False,
                collate_fn=self.collate_fn_pad,
            ),
            DataLoader(
                self.test_first_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=False,
                collate_fn=self.collate_fn_pad,
            ),
            DataLoader(
                self.test_last_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=False,
                collate_fn=self.collate_fn_pad,
            ),
        ]

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.collate_fn_pad,
        )


def get_data(cfg: DictConfig, tokenizer, num_bins):
    print("Loading dataset...")
    train_file = to_absolute_path(os.path.join(cfg.data.datapath, cfg.data.train_file))
    val_file = to_absolute_path(os.path.join(cfg.data.datapath, cfg.data.val_file))
    test_file = to_absolute_path(
        os.path.join(cfg.data.datapath, cfg.data.val_target_file)
    )

    hf_dataset = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "val": val_file,
            "test": test_file,
        },
    )
    print(
        f"Dataset loaded. Sizes - Train: {len(hf_dataset['train'])}, Val: {len(hf_dataset['val'])}, Test: {len(hf_dataset['test'])}"
    )

    # Calculate bin ranges
    bin_size = 4096 // num_bins
    bins = [(i * bin_size, (i + 1) * bin_size - 1) for i in range(num_bins)]
    print(f"Created {num_bins} bins: {bins}")

    # Initialize dataset splits
    dataset_splits = {
        "train": [],
        "val": [],
        "test_all": [],
        "test_first": [],
        "test_last": [],
    }

    samples_needed = {
        "train": int(cfg.data["num_train"]),
        "val": int(cfg.data["num_val"]),
        "test_all": int(cfg.eval["num_examples"]),
    }
    test_first_target = 256
    test_last_target = 256
    print(
        f"Samples needed - Train: {samples_needed['train']}, Val: {samples_needed['val']}, Test_all: {samples_needed['test_all']}"
    )

    def get_bin_idx(seq_len):
        for idx, (start, end) in enumerate(bins):
            if start <= seq_len <= end:
                return idx
        return num_bins - 1

    # Process test splits
    print("\nProcessing test splits...")
    test_all = []
    test_first = []
    test_last = []

    for example in hf_dataset["test"]:
        # Tokenize and process example
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
        input_ids = outputs["input_ids"]

        # Calculate sequence length and bin
        try:
            seq_len = input_ids.index(tokenizer.eos_token_id) + 1
        except ValueError:
            seq_len = len(input_ids)
        seq_len = min(seq_len, 4096)
        bin_idx = get_bin_idx(seq_len)

        # Assign to splits
        if len(test_all) < samples_needed["test_all"]:
            test_all.append(input_ids)  # Add to test_all first
        else:
            # After test_all is full, collect test_first/test_last from remaining data
            if bin_idx == 0 and len(test_first) < test_first_target:
                test_first.append(input_ids)
            if bin_idx == num_bins - 1 and len(test_last) < test_last_target:
                test_last.append(input_ids)

        # Early exit if all test splits are filled
        if (
            len(test_all) >= samples_needed["test_all"]
            and len(test_first) >= test_first_target
            and len(test_last) >= test_last_target
        ):
            break

    # Assign collected data to splits
    dataset_splits["test_all"] = test_all
    dataset_splits["test_first"] = test_first[:test_first_target]  # Cap at target size
    dataset_splits["test_last"] = test_last[:test_last_target]

    print(
        f"Test splits collected - all: {len(test_all)}, first: {len(test_first)}, last: {len(test_last)}"
    )

    print("\nProcessing train split with bin balancing (excluding first/last bins)...")
    train_target = samples_needed["train"]
    active_bins = num_bins - 2  # Exclude first and last bins
    per_bin, remainder = divmod(train_target, active_bins)
    bin_limits = {}
    bin_counts = defaultdict(int)

    # Initialize limits only for middle bins (1 to num_bins-2)
    for bin_idx in range(1, num_bins - 1):
        bin_limits[bin_idx] = per_bin + 1 if bin_idx <= remainder else per_bin

    total_collected = 0
    for example in hf_dataset["train"]:
        if total_collected >= train_target:
            break

        # Tokenize and bin calculation
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
        input_ids = outputs["input_ids"]

        try:
            seq_len = input_ids.index(tokenizer.eos_token_id) + 1
        except ValueError:
            seq_len = len(input_ids)
        seq_len = min(seq_len, 4096)
        bin_idx = get_bin_idx(seq_len)

        # Only consider middle bins (1 to num_bins-2)
        if 1 <= bin_idx < num_bins - 1:
            if bin_limits.get(bin_idx, 0) > 0:
                dataset_splits["train"].append(input_ids)
                bin_limits[bin_idx] -= 1
                bin_counts[bin_idx] += 1
                total_collected += 1

        if total_collected % 1000 == 0:
            print(f"Collected {total_collected}/{train_target} samples")

    # Process val split (existing logic)
    print("\nProcessing val split...")
    examples_collected = 0
    for example in hf_dataset["val"]:
        if examples_collected >= samples_needed["val"]:
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
        dataset_splits["val"].append(outputs["input_ids"])
        examples_collected += 1

    # Create final dataset
    final_dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"input_ids": dataset_splits["train"]}),
            "val": Dataset.from_dict({"input_ids": dataset_splits["val"]}),
            "test_all": Dataset.from_dict({"input_ids": dataset_splits["test_all"]}),
            "test_first": Dataset.from_dict(
                {"input_ids": dataset_splits["test_first"]}
            ),
            "test_last": Dataset.from_dict({"input_ids": dataset_splits["test_last"]}),
        }
    )

    print("\nFinal dataset sizes:")
    for split in final_dataset:
        print(f"{split}: {len(final_dataset[split])}")

    print("\nTraining bin distribution:")
    for bin_idx in range(num_bins):
        print(f"Bin {bin_idx}: {bin_counts[bin_idx]} samples")

    return final_dataset


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
