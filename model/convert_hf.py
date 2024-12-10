import json
import os
from pathlib import Path
from pprint import pprint
from typing import Optional, Union
import torch
import hydra
from litgpt.scripts.convert_lit_checkpoint import convert_lit_checkpoint
from litgpt.utils import copy_config_files, auto_download_checkpoint
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="../config", config_name="config", version_base=None)
def convert_hf(cfg: DictConfig):

    out_dir = Path(cfg.convert_hf.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(cfg.convert_hf.in_path)
    model_path = out_dir / "pytorch_model.bin"
    model_path = Path(model_path)

    copy_config_files(source_dir=source_dir, out_dir=out_dir)
    convert_lit_checkpoint(checkpoint_dir=source_dir, output_dir=out_dir)
    state_dict = torch.load(out_dir / "model.pth")
    torch.save(state_dict, model_path)

if __name__ == "__main__":
    convert_hf()