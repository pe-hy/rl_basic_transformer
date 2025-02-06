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

    def setup(self, stage=None):
        self.train_dataset = self.dataset["train"]
        self.val_dataset = self.dataset["val"]
        self.test_dataset = self.dataset["test"]

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
    test_file = to_absolute_path(os.path.join(cfg.data.datapath, cfg.data.test_file))

    hf_dataset = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "val": val_file,
            "test": test_file,
        },
    )

    # hf_dataset["train"] = hf_dataset["train"].select(range(int(1000)))

    def tokenize(examples):
        texts = [
            tokenizer.bos_token + ex + tokenizer.eos_token for ex in examples["text"]
        ]
        outputs = tokenizer(
            texts,
            truncation=True,
            max_length=cfg.model.block_size,
            padding="max_length",
            return_overflowing_tokens=False,
        )
        return {"input_ids": outputs["input_ids"]}

    tokenized_dataset = hf_dataset.map(
        tokenize, batched=True, remove_columns=hf_dataset["train"].column_names
    )

    return tokenized_dataset


def get_tokenizer(tok_data: DictConfig, for_filter: Optional[bool] = False):
    if for_filter:
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=to_absolute_path(f"../{tok_data.tokenizer_path}")
        )
    else:
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
