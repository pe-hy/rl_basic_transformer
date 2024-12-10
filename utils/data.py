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
        self.train_dataset = self.dataset['train']
        self.val_dataset = self.dataset['val']
        self.test_dataset = self.dataset['test']

    def collate_fn_pad(self,batch):
        x, y = zip(*batch)
        # Pad sequences to the maximum length in the batch
        x_padded = pad_sequence(x, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)
        return x_padded, y_padded
    
    def connect(
        self, max_seq_length: Optional[int] = None
    ) -> None:
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=self.num_workers,
                                drop_last=False,
                                collate_fn=self.collate_fn_pad)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=False,
                          collate_fn=self.collate_fn_pad)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=False,
                          collate_fn=self.collate_fn_pad)

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

def get_tokenizer(tok_data: DictConfig):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=to_absolute_path(tok_data.tokenizer_path))
    tokenizer.eos_token = "[EOS]"
    tokenizer.unk_token = "[UNK]"   
    tokenizer.pad_token = "[PAD]"
    tokenizer.mask_token = "[MASK]"
    tokenizer.bos_token = "[BOS]"
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer