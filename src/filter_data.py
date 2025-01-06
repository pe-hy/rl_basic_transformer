import hydra
from omegaconf import DictConfig
from datasets import load_dataset
import torch
from loguru import logger
import numpy as np
from hydra.utils import get_original_cwd, to_absolute_path
import os
import json
from transformers import PreTrainedTokenizerFast


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


def get_data(cfg: DictConfig, tokenizer):
    train_file = to_absolute_path(cfg.tok_data.train_file)
    val_file = to_absolute_path(cfg.tok_data.val_file)

    hf_dataset = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "val": val_file,
            "test": val_file,
        },
    )

    def tokenize(element):
        text = [
            tokenizer.bos_token
            + element["search_path"][e].strip()
            + tokenizer.eos_token
            for e in range(len(element["search_path"]))
        ]
        outputs = tokenizer(
            text,
            truncation=False,
            stride=0,
            return_length=True,
        )
        return {"length": outputs["length"]}

    tokenized_dataset = hf_dataset.map(tokenize, batched=True)
    return tokenized_dataset, hf_dataset


def get_data_with_filtering(cfg: DictConfig, tokenizer):
    tokenized_dataset, original_dataset = get_data(cfg, tokenizer)
    stats = {}
    filtered_datasets = {}

    output_dir = to_absolute_path("../data/sos_filtered")
    os.makedirs(output_dir, exist_ok=True)

    for split in tokenized_dataset.keys():
        logger.info(f"\nProcessing {split} split...")

        original_count = len(tokenized_dataset[split])
        lengths = tokenized_dataset[split]["length"]

        valid_indices = [
            i for i, length in enumerate(lengths) if length <= cfg.model.block_size
        ]

        filtered_data = [original_dataset[split][idx] for idx in valid_indices]

        output_file = f"{split}1_b4_t30_n500000_dfs_filtered.json"
        output_path = os.path.join(output_dir, output_file)
        logger.info(f"Saving filtered {split} data to {output_path}")

        with open(output_path, "w") as f:
            for item in filtered_data:
                json.dump(item, f)
                f.write("\n")

        logger.info(f"Saved filtered {split} data with {len(filtered_data)} examples")

        filtered_count = original_count - len(valid_indices)
        filtered_percentage = (filtered_count / original_count) * 100

        stats[split] = {
            "original_count": original_count,
            "filtered_count": filtered_count,
            "remaining_count": len(valid_indices),
            "filtered_percentage": filtered_percentage,
            "max_length": max(lengths),
            "min_length": min(lengths),
            "avg_length": np.mean(lengths).item(),
            "std_length": np.std(lengths).item(),
        }

        logger.info(f"Split: {split}")
        logger.info(f"Original samples: {original_count:,}")
        logger.info(f"Filtered out: {filtered_count:,} ({filtered_percentage:.2f}%)")
        logger.info(f"Remaining: {len(valid_indices):,}")
        logger.info("Length statistics:")
        logger.info(f"  Max: {stats[split]['max_length']:,}")
        logger.info(f"  Min: {stats[split]['min_length']:,}")
        logger.info(f"  Avg: {stats[split]['avg_length']:.2f}")
        logger.info(f"  Std: {stats[split]['std_length']:.2f}")

    return filtered_datasets, stats


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    tokenizer = get_tokenizer(cfg.tok_data)
    filtered_datasets, stats = get_data_with_filtering(cfg, tokenizer)
    return filtered_datasets, stats


if __name__ == "__main__":
    main()
