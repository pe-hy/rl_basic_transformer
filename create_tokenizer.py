from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast
from datasets import Dataset, DatasetDict
from transformers import DataCollatorForLanguageModeling
from torch.utils.data.dataloader import DataLoader
import torch
import pickle
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    vocab = get_vocab(cfg)
    tokenizer = get_tokenizer(vocab, cfg)


def get_tokenizer(vocab, cfg):
    vocab = {s: i for i, s in enumerate(vocab)}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.add_special_tokens(["[BOS]", "[PAD]", "[MASK]", "[UNK]", "[EOS]"])
    tokenizer.save("tokenizer/" + "tokenizer.json")
    print("tokenizer saved to:", "tokenizer/" + "tokenizer.json")
    # tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"data/tokenizer.json")
    # tokenizer.eos_token = "[SEP]"
    # tokenizer.unk_token = "[UNK]"
    # tokenizer.pad_token = "[PAD]"
    # tokenizer.mask_token = "[MASK]"
    return tokenizer


def get_vocab(cfg: DictConfig):
    with open(cfg.data.datapath_train, "rb") as f:
        train = pickle.load(f)

    with open(cfg.data.datapath_val, "rb") as f:
        val = pickle.load(f)


    # Initialize vocabulary set
    vocab = set()
    
    # Process train data
    for idx in train:
        tokens = train[idx]['tokens']
        vocab.update(tokens)
        
    # Process val data
    for idx in val:
        tokens = val[idx]['tokens']
        vocab.update(tokens)
        
    print(vocab)
    print("Number of unique tokens:", len(vocab))
    return vocab


if __name__ == "__main__":
    main()