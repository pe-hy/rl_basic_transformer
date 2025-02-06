from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast
from datasets import Dataset, DatasetDict
from transformers import DataCollatorForLanguageModeling
from torch.utils.data.dataloader import DataLoader
import torch
import json
import hydra
from omegaconf import DictConfig, OmegaConf

import os


@hydra.main(
    config_path="../config", config_name="config_pythia_cylinder", version_base=None
)
def main(cfg: DictConfig):
    vocab = get_vocab(cfg)
    tokenizer = get_tokenizer(vocab, cfg)


def get_tokenizer(vocab, cfg):
    vocab = {s: i for i, s in enumerate(vocab)}
    # Initialize tokenizer with complete vocabulary
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.add_special_tokens(["[BOS]", "[PAD]", "[MASK]", "[UNK]", "[EOS]"])
    # Save tokenizer
    save_path = f"../{cfg.tok_data.tokenizer_path}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    print("tokenizer saved to:", save_path)

    return tokenizer


def get_vocab(cfg: DictConfig):
    with open(cfg.tok_data.train_file, "rb") as f:
        train = json.load(f)

    with open(cfg.tok_data.val_file, "rb") as f:
        val = json.load(f)

    with open(cfg.tok_data.test_file, "rb") as f:
        test = json.load(f)

    data = train + val + test
    data = [i for i in data]
    data = " ".join(data)
    vocab = set(data.split())
    print("Num of tokens:", len(vocab))
    return vocab


if __name__ == "__main__":
    main()
