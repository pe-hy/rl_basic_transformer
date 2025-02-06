import json
import os
from tokenizers import Tokenizer
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from data_pythia import get_tokenizer


def filter_by_length(data: list, tokenizer: Tokenizer, max_length: int) -> list:
    """Filter examples that exceed max token length."""
    filtered_data = []
    removed_count = 0
    max_found_length = 0

    print(f"\nFiltering examples longer than {max_length} tokens...")
    for example in tqdm(data):
        tokens = tokenizer(example)["input_ids"]
        token_length = len(tokens)
        max_found_length = max(max_found_length, token_length)

        if token_length <= max_length:
            filtered_data.append(example)
        else:
            removed_count += 1

    return filtered_data, removed_count, max_found_length


def process_and_save_file(file_path: str, tokenizer: Tokenizer, max_length: int):
    """Process a single JSON file and overwrite with filtered data."""
    print(f"\nProcessing {file_path}")

    # Load data
    with open(file_path, "r") as f:
        data = json.load(f)

    original_count = len(data)
    print(f"Original example count: {original_count}")

    # Filter data
    filtered_data, removed_count, max_length_found = filter_by_length(
        data, tokenizer, max_length
    )

    # Save filtered data
    with open(file_path, "w") as f:
        json.dump(filtered_data, f, indent=2)

    print(f"Results for {os.path.basename(file_path)}:")
    print(f"- Examples removed: {removed_count}")
    print(f"- Maximum token length found: {max_length_found}")
    print(f"- Final example count: {len(filtered_data)}")
    print(f"- Removal percentage: {(removed_count/original_count)*100:.2f}%")

    return removed_count, max_length_found


@hydra.main(
    config_path="../config", config_name="config_pythia_cylinder", version_base=None
)
def main(cfg: DictConfig):
    # Load tokenizer
    tokenizer = get_tokenizer(cfg.tok_data, for_filter=True)

    # Process all splits
    total_removed = 0
    overall_max_length = 0

    files_to_process = [
        cfg.tok_data.train_file,
        cfg.tok_data.val_file,
        cfg.tok_data.test_file,
    ]

    for file_path in files_to_process:
        removed, max_length = process_and_save_file(
            file_path, tokenizer, max_length=4096
        )
        total_removed += removed
        overall_max_length = max(overall_max_length, max_length)

    print("\nOverall Statistics:")
    print(f"Total examples removed: {total_removed}")
    print(f"Maximum token length found across all splits: {overall_max_length}")


if __name__ == "__main__":
    main()
