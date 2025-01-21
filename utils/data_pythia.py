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
    """
    Creates curriculum datasets with proper stage handling while preserving test data.
    """
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

    for split in hf_dataset:
        hf_dataset[split] = hf_dataset[split].add_column(
            "split", [split] * len(hf_dataset[split])
        )

    num_stages = 10
    curriculum_datasets = []

    def create_tokenize_function(curr_stage):
        def tokenize_for_stage(examples):
            processed_examples = {"input_ids": []}

            for i in range(len(examples["search_path"])):
                if examples["split"][i] == "test":
                    # For test split, always use the complete path
                    text = (
                        tokenizer.bos_token
                        + examples["search_path"][i].strip()
                        + tokenizer.eos_token
                    )
                else:
                    # For train and val, apply curriculum
                    splits = split_search_path([examples["search_path"][i]], curr_stage)
                    text = tokenizer.bos_token + splits[0].strip() + tokenizer.eos_token

                outputs = tokenizer(
                    text,
                    truncation=True,
                    max_length=cfg.model.block_size,
                    return_overflowing_tokens=True,
                    return_length=True,
                    stride=0,
                    padding="max_length",
                )
                processed_examples["input_ids"].append(outputs["input_ids"][0])

            return processed_examples

        return tokenize_for_stage

    for stage in range(num_stages):
        # print(f"\nProcessing stage {stage}")
        tokenize_function = create_tokenize_function(stage)

        tokenized_dataset = hf_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=hf_dataset["train"].column_names,
        )

        curriculum_datasets.append(tokenized_dataset)

        # Debug: print example from train and test for comparison
        # if len(tokenized_dataset["train"]) > 0:
        #     print(f"\nStage {stage} train example:")
        #     print(tokenizer.decode(tokenized_dataset["train"][0]["input_ids"]))
        #     if len(tokenized_dataset["test"]) > 0:
        #         print(f"\nStage {stage} test example (should be complete):")
        #         print(tokenizer.decode(tokenized_dataset["test"][0]["input_ids"]))

    return curriculum_datasets


def split_search_path(path_list, stage, num_stages=10):
    """
    Splits search paths for curriculum learning.
    """
    all_splits = []
    for path in path_list:
        # Get comma positions
        comma_positions = [i for i, char in enumerate(path) if char == ","]

        if stage == num_stages - 1:
            split_path = path
        elif not comma_positions:
            split_path = path
        else:
            if stage == 0:
                split_position = comma_positions[0]
            else:
                num_commas = max(
                    1, int((stage + 1) * len(comma_positions) / num_stages)
                )
                split_position = comma_positions[num_commas - 1]
            split_path = path[:split_position]

        all_splits.append(split_path)

    return all_splits


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
