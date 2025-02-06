import pickle
import json
import random
import os
from typing import List
from omegaconf import DictConfig, OmegaConf
import hydra


def process_examples(examples_list):
    # Initialize list to store processed logs
    processed_logs = []

    # Iterate through each tuple in the list of examples
    for example_tuple in examples_list:
        # Get third element (list of strings/log entries)
        log_entries = example_tuple[2]

        # Process this list: trim each entry and join with semicolons
        processed_log = " ; ".join(entry.strip() for entry in log_entries)

        # Add to results
        processed_logs.append(processed_log)

    return processed_logs


def check_overlap(set1, set2, set1_name, set2_name, data1, data2):
    overlap = set1.intersection(set2)
    if overlap:
        overlap_example = next(iter(overlap))  # Get first overlapping example
        print(f"\nOverlap found between {set1_name} and {set2_name}!")
        print(f"Overlapping example: {overlap_example}")
        print(f"Found in {set1_name} at index: {data1.index(overlap_example)}")
        print(f"Found in {set2_name} at index: {data2.index(overlap_example)}")
        raise AssertionError(f"Data leakage between {set1_name} and {set2_name}")


def split_and_export_data(cfg, processed_examples: List[str]):
    # Convert to list if not already (in case it's a generator)
    examples = list(processed_examples)
    print(f"Original number of examples: {len(examples)}")

    # Remove duplicates while preserving order
    examples = list(dict.fromkeys(examples))
    print(f"Number of examples after removing duplicates: {len(examples)}")

    # Shuffle the dataset after removing duplicates
    random.shuffle(examples)

    # Split into test, val, and train
    test_data = examples[: cfg.data_preprocess.num_test]
    val_data = examples[
        cfg.data_preprocess.num_test : cfg.data_preprocess.num_test
        + cfg.data_preprocess.num_val
    ]
    train_data = examples[cfg.data_preprocess.num_test + cfg.data_preprocess.num_val :]

    # Check for overlaps with detailed error reporting
    test_set = set(test_data)
    val_set = set(val_data)
    train_set = set(train_data)

    check_overlap(test_set, train_set, "test", "train", test_data, train_data)
    check_overlap(val_set, train_set, "validation", "train", val_data, train_data)
    check_overlap(test_set, val_set, "test", "validation", test_data, val_data)

    # Shuffle train data again
    random.shuffle(train_data)

    # Create cylinder directory if it doesn't exist
    os.makedirs(os.path.dirname(cfg.tok_data.train_file), exist_ok=True)

    # Export to JSON files
    splits = {
        f"{cfg.tok_data.test_file}": test_data,
        f"{cfg.tok_data.val_file}": val_data,
        f"{cfg.tok_data.train_file}": train_data,
    }

    for filename, data in splits.items():
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    # Print some statistics
    print(f"Exported:")
    print(f"- Test set: {len(test_data)} examples")
    print(f"- Validation set: {len(val_data)} examples")
    print(f"- Training set: {len(train_data)} examples")


@hydra.main(
    config_path="../config", config_name="config_pythia_cylinder", version_base=None
)
def main(cfg: DictConfig):
    with open(f"{cfg.data_preprocess.raw_data_path}", "rb") as f:
        d = pickle.load(f)

    processed_examples = process_examples(d)
    split_and_export_data(cfg, processed_examples)


if __name__ == "__main__":
    main()
