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

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Datamodule(LightningDataModule):
    def __init__(self, dataset, batch_size, num_workers, tokenizer):
        super(Datamodule, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
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
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.collate_fn_pad,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.collate_fn_pad,
        )


def get_data(cfg: DictConfig, tokenizer):
    train_file = to_absolute_path(os.path.join(cfg.data.datapath, cfg.data.train_file))
    val_file = to_absolute_path(os.path.join(cfg.data.datapath, cfg.data.val_file))
    # val_target_file = os.path.join(data.data_dir, data.val_target_file)

    hf_dataset = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "val": val_file,
            "test": val_file,
        },
    )

    hf_dataset["train"] = hf_dataset["train"].select(range(int(cfg.data["num_train"])))
    hf_dataset["val"] = hf_dataset["val"].select(range(int(cfg.data["num_val"])))
    hf_dataset["test"] = hf_dataset["test"].select(range(int(cfg.eval["num_examples"])))

    def tokenize(element):
        text = [
            tokenizer.bos_token
            + element["search_path"][e].strip()
            + tokenizer.eos_token
            for e in range(len(element["search_path"]))
        ]
        outputs = tokenizer(
            text,
            truncation=True,
            max_length=cfg.model.block_size,
            return_overflowing_tokens=True,
            return_length=True,
            stride=0,
            padding="max_length",
        )
        return {"input_ids": outputs["input_ids"]}

    tokenized_dataset = hf_dataset.map(
        tokenize, batched=True, remove_columns=hf_dataset["train"].column_names
    )

    return tokenized_dataset


def get_curriculum(cfg: DictConfig, tokenizer):
    train_file = to_absolute_path(os.path.join(cfg.data.datapath, cfg.data.train_file))
    val_file = to_absolute_path(os.path.join(cfg.data.datapath, cfg.data.val_file))
    val_target_file = os.path.join(cfg.data.datapath, cfg.data.val_target_file)

    hf_dataset = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "val": val_file,
            "test": val_target_file,
        },
    )

    hf_dataset["train"] = hf_dataset["train"].select(range(int(cfg.data["num_train"])))
    hf_dataset["val"] = hf_dataset["val"].select(range(int(cfg.data["num_val"])))
    hf_dataset["test"] = hf_dataset["test"].select(range(int(cfg.eval["num_examples"])))

    def split_search_path(path_list, stage, num_stages=10):
        all_splits = []
        for path in path_list:
            if stage == num_stages - 1:  # If it's the final stage (10th)
                all_splits.append(path)  # Keep the original full path
            else:
                # Get all comma positions
                comma_positions = [i for i, char in enumerate(path) if char == ","]

                if not comma_positions:
                    # If no commas, just return the whole path
                    all_splits.append(path)
                    continue

                # Calculate desired split point based on total path length
                total_length = len(path)
                target_position = int(total_length * ((stage + 1) / 10))

                # Find the appropriate comma position:
                # If target position is before first comma, use first comma
                # If target position is after a comma, use the last comma before target
                if target_position <= comma_positions[0]:
                    split_position = comma_positions[0]
                else:
                    # Get all commas before target position
                    earlier_commas = [
                        pos for pos in comma_positions if pos <= target_position
                    ]
                    if earlier_commas:
                        split_position = max(earlier_commas)
                    else:
                        split_position = comma_positions[0]

                split_path = path[:split_position]  # Split before the comma
                all_splits.append(split_path)

        return all_splits

    def tokenize_for_curriculum(element, stage=None):
        if stage is not None and (
            element["split"] == "train" or element["split"] == "val"
        ):
            # For train and val splits, process according to the curriculum
            text = []
            for path_idx in range(len(element["search_path"])):
                splits = split_search_path([element["search_path"][path_idx]], stage)
                processed_text = (
                    tokenizer.bos_token + splits[0].strip() + tokenizer.eos_token
                )
                text.append(processed_text)
        else:
            # For test split or when stage is None, process normally
            text = [
                tokenizer.bos_token
                + element["search_path"][e].strip()
                + tokenizer.eos_token
                for e in range(len(element["search_path"]))
            ]

        outputs = tokenizer(
            text,
            truncation=True,
            max_length=cfg.model.block_size,
            return_overflowing_tokens=True,
            return_length=True,
            stride=0,
            padding="max_length",
        )
        return {"input_ids": outputs["input_ids"]}

    # Add split information to the datasets
    for split in hf_dataset:
        hf_dataset[split] = hf_dataset[split].add_column(
            "split", [split] * len(hf_dataset[split])
        )

    # Create exactly 10 curriculum datasets
    num_stages = 10
    curriculum_datasets = []
    for i in range(num_stages):
        tokenized_dataset = hf_dataset.map(
            lambda x: tokenize_for_curriculum(x, i),
            batched=True,
            remove_columns=hf_dataset["train"].column_names,
        )
        curriculum_datasets.append(tokenized_dataset)

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
