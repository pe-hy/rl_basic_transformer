import os
import json
import hydra
from omegaconf import OmegaConf, DictConfig
from transformers import AutoTokenizer
def update_tokenizer(cfg):
    full_path = cfg.dl_model.full_path
    config_path = os.path.join(full_path, "tokenizer_config.json")

    updated_config = {
        "add_prefix_space": False,
        "add_bos_token": True,
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
        "tokenizer_class": "GPTNeoXTokenizer",
        "unk_token": "[UNK]"
    }

    with open(config_path, 'w') as f:
        json.dump(updated_config, f, indent=2)

    print(f"Updated tokenizer configuration saved to {config_path}")

def update_config_vocab_sizes(cfg):
    # Construct the full path to the model folder
    full_path = os.path.join(cfg.dl_model.model_folder, cfg.dl_model.name)
    
    # Paths to config files
    config_json_path = os.path.join(full_path, "config.json")
    model_config_yaml_path = os.path.join(full_path, "model_config.yaml")
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(full_path)
    
    # Get vocab size
    vocab_size = len(tokenizer)

    # Get blocksize
    block_size = cfg.model.block_size

    # Update JSON config
    with open(config_json_path, 'r') as f:
        config_json = json.load(f)
    # config_json['vocab_size'] = vocab_size
    config_json['max_position_embeddings'] = block_size
    config_json['bos_token_id'] = tokenizer.bos_token_id
    config_json['eos_token_id'] = tokenizer.eos_token_id
    with open(config_json_path, 'w') as f:
        json.dump(config_json, f, indent=2)
    print(f"Updated vocab_size to {vocab_size} in {config_json_path}")
    print(f"Updated max_positional_embeddings to {block_size} in {config_json_path}")

    # Update YAML config
    model_config = OmegaConf.load(model_config_yaml_path)
    model_config.vocab_size = vocab_size
    model_config.block_size = block_size
    OmegaConf.save(model_config, model_config_yaml_path)
    print(f"Updated vocab_size to {vocab_size} in {model_config_yaml_path}")
    print(f"Updated block_size to {block_size} in {model_config_yaml_path}")

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    update_tokenizer(cfg)
    update_config_vocab_sizes(cfg)

if __name__ == "__main__":
    main()