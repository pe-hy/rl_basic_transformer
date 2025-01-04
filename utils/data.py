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
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        self.return_prediction_mask = True

    def setup(self, stage=None):
        self.train_dataset = self.dataset["train"]
        self.val_dataset = self.dataset["val"]
        self.test_dataset = self.dataset["test"]

    def custom_collate_fn(self, batch):
        torch.set_printoptions(threshold=10_000)
        # batch is a list of lists
        batch = [i["input_ids"] for i in batch]
        batch = [torch.tensor(i) for i in batch]  # convert to tensor
        batch = pad_sequence(
            batch, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attn = torch.ones_like(batch)
        attn = attn.masked_fill(batch == self.tokenizer.pad_token_id, 0)
        batch_dict = {"input_ids": batch, "attention_mask": attn}
        _, prefix_attn = self._eval_get_prefix(batch_dict)
        # print("prefix_attn", prefix_attn[1])
        # print("attn[1]: ", attn[1])
        if self.return_prediction_mask:
            batch_dict["prediction_mask"] = attn - prefix_attn
            # print("pred_mask: ", batch_dict["prediction_mask"][1])
        return batch_dict

    def _eval_get_prefix(self, x):
        """This function is filtering inputs up to the `<PATH_START>` token and then padding the rest of the input. Example:"""
        # setup
        pad_id = self.tokenizer.pad_token_id
        separator = self.tokenizer.encode(",")[0]
        prefix = x["input_ids"].clone()
        first_separator_pos = (prefix == separator).float().argmax(dim=1)
        # values = torch.tensor(
        #     [prefix[i, first_separator_pos[i]] for i in range(prefix.size(0))]
        # )
        attn = x["attention_mask"].clone()
        # filter inputs up to seprator then pad
        up_to_sep_mask = torch.zeros_like(prefix, dtype=torch.bool)
        batch_indices = torch.arange(prefix.size(0), device=prefix.device)
        up_to_sep_mask[batch_indices, first_separator_pos] = True
        # up_to_sep_mask = prefix == separator
        # include the separator in the up_to_sep_mask
        up_to_sep_mask = up_to_sep_mask.cumsum(dim=1) > 0
        up_to_sep_mask[batch_indices, first_separator_pos] = False  # keep sep in prefix
        # replace suffix with padding
        prefix = prefix.masked_fill(up_to_sep_mask, pad_id)
        # mask attn for suffix
        attn = attn.masked_fill(up_to_sep_mask, 0)
        return prefix, attn

    def _eval_get_model_answers(self, prefix, attn, model, **model_kwargs):
        """This function is generating model answers for a given set of input prefixes
        takes a tensor which contains the prefix and padding which will be replaced by the model answers
        attn ensures the model doesn't attend to the padding tokens (this is important for bi-directional models)
        example:
        prefix = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 0, 0]])
        attn = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]])
        the model will make predictions for each token at every iteration but will
        only replace the 0s when the iteration index reaches them
        """
        eot_token = self.tokenizer.encode(".")[0]
        ans = prefix.clone()
        attn = attn.clone()
        column = range(prefix.shape[0])
        for idx in range(ans.shape[1]):
            print(idx)
            # if the ans is masked replaced with model output
            old_values = ans[column, idx]
            column_attn = attn[column, idx]
            # skip preds if only old values would be chosen anyway
            if not column_attn.all():
                preds = model(
                    input_ids=ans, attention_mask=attn, **model_kwargs
                ).logits.argmax(dim=-1)
                # keep old values if they are not attn masked
                ans[column, idx] = old_values.where(
                    column_attn != 0, preds[column, idx]
                )
            # update attn mask with 1s for each non eot token
            attn[column, idx] = attn[column, idx].masked_fill(
                ans[column, idx] != eot_token, 1
            )

        return ans

    def eval_fn(
        self, model, batch, dataloader_idx=0, return_samples=False, **model_kwargs
    ):
        """runs some evaluation based on which data_loader. returns a dict containing
        at least the 'metrics' key whose value is a dict of metrics.
        can also return keys for generated text samples"""
        # get accuracies based on the model's output
        # 1. find prefix to condition the model
        # 2. model.generate(based on prefix)
        # 3. check per token accuracy for the suffix
        # val_set_name = list(self.val.keys())[dataloader_idx]
        # val_set_name = "val"
        prefix, prefix_attn = self._eval_get_prefix(batch)
        ans = self._eval_get_model_answers(prefix, prefix_attn, model, **model_kwargs)
        attn = batch["attention_mask"]
        # suffix_attn = attn.clone()
        # # convert to 1 for each token in the suffix
        # suffix_attn = suffix_attn.where(prefix_attn == 0, 0)
        res = {
            "metrics": {
                f"acc": 0,
                f"full_acc": 0,
            }
        }
        return res

    def connect(self, max_seq_length: Optional[int] = None) -> None:
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.custom_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.custom_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.custom_collate_fn,
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
