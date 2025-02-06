import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from transformers import TrainerCallback
from lightning.pytorch.callbacks import Callback
import os
import json
import torch
import numpy as np
import wandb
import pandas as pd
from datetime import datetime
from tqdm import trange
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM
from litgpt.scripts.convert_lit_checkpoint import convert_lit_checkpoint
from litgpt.utils import copy_config_files, auto_download_checkpoint
import torch
from pathlib import Path
from datetime import datetime


def convert_litgpt_to_hf(cfg):

    out_dir = Path(cfg.convert_hf.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(cfg.convert_hf.in_path)
    model_path = out_dir / "pytorch_model.bin"
    model_path = Path(model_path)

    copy_config_files(source_dir=source_dir, out_dir=out_dir)
    convert_lit_checkpoint(checkpoint_dir=source_dir, output_dir=out_dir)

    state_dict = torch.load(out_dir / "model.pth")
    torch.save(state_dict, model_path)
    hf_model = AutoModelForCausalLM.from_pretrained(
        out_dir,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        state_dict=state_dict,
        attn_implementation="flash_attention_2",
    )
    return hf_model


class CylinderEvaluator:
    def __init__(self, config, test_set, tokenizer, step=None, model=None):
        self.config = config
        self.num_examples = config.eval.num_examples
        self.batch_size = config.eval.batch_size
        self.global_step = step
        self.tokenizer = tokenizer
        self.results_dir = config.eval.results_dir
        self.model = model
        self.hf_model = convert_litgpt_to_hf(config)
        self.test_set = test_set
        self.step = step
        os.makedirs(self.results_dir, exist_ok=True)

        self.prompts = self.get_prompts()

    def get_prompts(self):
        search_token_id = self.tokenizer.encode("Search:", add_special_tokens=False)[0]

        prompts = []
        for sample in self.test_set:
            input_ids = sample["input_ids"]
            split_index = input_ids.index(search_token_id)
            # Take everything up to Search: token
            prompt_ids = input_ids[: split_index + 1]

            # Decode to text, add BOS token at start
            prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
            full_prompt = self.tokenizer.bos_token + prompt_text

            # Re-encode with BOS token
            prompt_with_bos = self.tokenizer.encode(
                full_prompt, add_special_tokens=False
            )
            prompts.append(prompt_with_bos)

        return prompts

    def get_preds(self):
        batch_size = self.batch_size
        data = self.prompts
        tokenizer = self.tokenizer
        output_texts_concat = []

        self.hf_model.cuda()
        self.hf_model.eval()

        for b in trange(0, len(data), batch_size):
            batch = data[b : min(b + batch_size, len(data))]
            batch_text = [tokenizer.decode(x, skip_special_tokens=False) for x in batch]
            tokenizer.padding_side = "left"
            inputs = tokenizer(batch_text, return_tensors="pt", padding=True).to("cuda")
            input_prompt = inputs["input_ids"]
            output_texts = ["" for _ in range(len(batch))]

            outputs = self.hf_model.generate(
                input_ids=input_prompt,
                pad_token_id=tokenizer.pad_token_id,
                attention_mask=inputs["attention_mask"].to("cuda"),
                max_length=self.config.model.block_size,
                num_beams=1,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )

            output_text = tokenizer.batch_decode(outputs, skip_special_tokens=False)

            output_texts = [
                ot + ot_now for ot, ot_now in zip(output_texts, output_text)
            ]
            output_texts_concat += output_texts

        return output_texts_concat

    def save(self, predictions, reasons):
        eval_dir = os.path.join(self.config.eval.results_dir, f"step_{self.step}")
        os.makedirs(eval_dir, exist_ok=True)
        results_file = os.path.join(eval_dir, f"results_{self.num_examples}.json")
        with open(results_file, "w") as f:
            json.dump(
                {"predictions": predictions, "reasons": reasons},
                f,
                indent=4,
            )

    def evaluate(self):
        preds = self.get_preds()
        reasons = []
        for pred in preds:
            try:
                reasons.append(parse_and_validate(pred))
            except:
                reasons.append("Evaluator failure.", {pred})
        valid_results = reasons.count("valid")

        acc = valid_results / len(reasons)

        self.save(preds, reasons)
        del self.hf_model
        torch.cuda.empty_cache()

        return acc


def parse_and_validate(input_string):
    import re

    # Crop the input string at the first occurrence of 'END'
    end_index = input_string.find("END")
    if end_index != -1:
        input_string = input_string[:end_index]

    # Helper function to convert bracketed content to a dictionary
    def parse_dict(data_str):
        entries = data_str.strip("[]").split(",")
        result_dict = {}
        for entry in entries:
            parts = entry.split(":")
            if len(parts) == 2:
                try:
                    key, value = int(parts[0].strip()), int(parts[1].strip())
                    result_dict[key] = value
                except ValueError:
                    continue  # Ignore malformed entries
        return result_dict

    # Extract Goal
    goal_match = re.search(r"Goal: \[(.*?)\]", input_string)
    if goal_match:
        goal = parse_dict(goal_match.group(1))
    else:
        return "Invalid input - Goal not found."

    # Find the last occurrence of 'CS' and the immediately following 'CG'
    cs_matches = list(re.finditer(r"CS: \[(.*?)\]", input_string))
    cg_matches = list(re.finditer(r"CG: \[(.*?)\]", input_string))

    if not cs_matches:
        return "Invalid input - CS not found."
    if not cg_matches:
        return "Invalid input - CG not found."

    # Get the last occurrences
    last_cs = cs_matches[-1]
    # Find the next CG after the last CS
    for match in cg_matches:
        if match.start() > last_cs.start():
            last_cg = match
            break
    else:
        return "Invalid input - CG not properly placed after last CS."

    # Parse the last CS and CG
    current_state = parse_dict(last_cs.group(1))
    change_goal = parse_dict(last_cg.group(1))

    # Update current state based on change goal
    for index, value in change_goal.items():
        if index in current_state:
            current_state[index] = value

    # Compare updated current state with the goal
    is_valid = current_state == goal

    return "valid" if is_valid else "invalid"
