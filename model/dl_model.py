import hydra
from litgpt import LLM
from litgpt.scripts import download
from omegaconf import DictConfig, OmegaConf
import json
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from pathlib import Path 
import shutil
@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    checkpoint_dir = Path(cfg.dl_model.model_folder)
    download.download_from_hub(repo_id=cfg.dl_model.name, tokenizer_only=False, checkpoint_dir=checkpoint_dir)
    llm = LLM.load("EleutherAI/pythia-160m", tokenizer_dir="EleutherAI/pythia-160m", init="random")
    llm.save("weights")
    del llm
    weights_dir = Path("weights")
    source_path = weights_dir / "lit_model.pth"
    target_dir = Path(cfg.dl_model.full_path)
    target_path = target_dir / "lit_model.pth"
    
    target_dir.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        target_path.unlink()
    shutil.move(str(source_path), str(target_path))
    shutil.rmtree(weights_dir)

    get_tokenizer(cfg)

def get_tokenizer(cfg: DictConfig):
    vocab = get_vocab(cfg)
    tokenizer = build_tokenizer(vocab, cfg)

def build_tokenizer(vocab, cfg):
    vocab = {s:i for i,s in enumerate(vocab)}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.add_special_tokens(['[BOS]', '[PAD]', '[MASK]', '[UNK]', '[EOS]'])
    tokenizer.save(cfg.dl_model.full_path + "/" + "tokenizer.json")
    print("tokenizer saved to:", cfg.dl_model.full_path + "tokenizer.json")
    return tokenizer

def get_vocab(cfg: DictConfig):
    with open(cfg.tok_data.train_file, "rb") as f:
        train = json.load(f)

    with open(cfg.tok_data.val_file, "rb") as f:
        val = json.load(f)

    with open(cfg.tok_data.val_target_file, "rb") as f:
        val_target = json.load(f)

    data = train + val + val_target
    data = [i["search_path"] for i in data]
    data = " ".join(data)
    vocab = set(data.split())
    print("Num of tokens:", len(vocab))
    return vocab

if __name__ == "__main__":
    main()