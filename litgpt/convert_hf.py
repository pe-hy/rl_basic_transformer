import json
import os
import time
from pathlib import Path
from pprint import pprint
from typing import Optional, Union
import torch

from litgpt.scripts.convert_lit_checkpoint import convert_lit_checkpoint
from litgpt.utils import copy_config_files, auto_download_checkpoint

def timed_conversion():
    # Start overall timer
    total_start_time = time.time()

    # Create output directory
    out_dir = Path("hf_trained_model")
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path("trained_model")
    model_path = out_dir / "pytorch_model.bin"
    model_path = Path(model_path)

    # Time config file copying
    config_start_time = time.time()
    copy_config_files(source_dir=source_dir, out_dir=out_dir)
    config_end_time = time.time()
    config_duration = config_end_time - config_start_time

    # Time checkpoint conversion
    convert_start_time = time.time()
    convert_lit_checkpoint(checkpoint_dir=source_dir, output_dir=out_dir)
    convert_end_time = time.time()
    convert_duration = convert_end_time - convert_start_time

    # Time state dict loading and saving
    load_save_start_time = time.time()
    state_dict = torch.load(out_dir / "model.pth")
    torch.save(state_dict, model_path)
    load_save_end_time = time.time()
    load_save_duration = load_save_end_time - load_save_start_time

    # Calculate total time
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    # Print timing results
    print("\n--- Conversion Timing Results ---")
    print(f"Config File Copying: {config_duration:.4f} seconds")
    print(f"LitGPT Checkpoint Conversion: {convert_duration:.4f} seconds")
    print(f"Model State Dict Loading/Saving: {load_save_duration:.4f} seconds")
    print(f"Total Conversion Time: {total_duration:.4f} seconds")

    return {
        "config_copying_time": config_duration,
        "checkpoint_conversion_time": convert_duration,
        "model_load_save_time": load_save_duration,
        "total_time": total_duration
    }

# Run the timed conversion
timing_results = timed_conversion()