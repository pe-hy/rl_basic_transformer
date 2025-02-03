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
from utils.countdown_utils import *
from tqdm import trange
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM
from litgpt.scripts.convert_lit_checkpoint import convert_lit_checkpoint
from litgpt.utils import copy_config_files, auto_download_checkpoint
import torch
from pathlib import Path
from datetime import datetime

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


class CountdownEvaluator:
    def __init__(self, config, tokenizer, step=None, model=None):
        self.config = config
        self.num_examples = config.eval.num_examples
        self.batch_size = config.eval.batch_size
        self.global_step = step
        self.tokenizer = tokenizer
        self.results_dir = config.eval.results_dir
        self.hf_model = None
        self.step = step

        os.makedirs(self.results_dir, exist_ok=True)

        # Initialize results DataFrame
        self.csv_path = os.path.join(self.results_dir, "eval_results.csv")
        if os.path.exists(self.csv_path):
            self.results_df = pd.read_csv(self.csv_path)
        else:
            self.results_df = pd.DataFrame(
                columns=[
                    "step",
                    "timestamp",
                    "average_rating",
                    "average_true_rating",
                    "accuracy",
                    "true_accuracy",
                    "predictions",
                ]
            )

    def eval_ll(
        self,
        model,
        tokenizer,
        data,
        batch_size=128,
        context_len=4096,
        temperature=0.0,
        n=1,
    ):
        output_texts_concat = []
        for b in trange(0, len(data), batch_size):
            batch = data[b : min(b + batch_size, len(data))]
            output_texts = ["" for _ in range(len(batch))]
            tokenizer.padding_side = "left"
            inputs = tokenizer(batch, return_tensors="pt", padding=True).to("cuda")
            inputs = inputs["input_ids"]

            if n == 1:
                outputs = model.generate(
                    input_ids=inputs,
                    pad_token_id=tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs),
                    max_length=context_len,
                    num_beams=1,
                    do_sample=False,
                )
                output_tokens = outputs
                output_text = tokenizer.batch_decode(
                    output_tokens, skip_special_tokens=False
                )
                tokenizer.padding_side = "left"
                output_texts = [
                    ot + ot_now for ot, ot_now in zip(output_texts, output_text)
                ]
                output_texts_concat += output_texts

        return output_texts_concat

    def evaluate_dataset(self, data, prefix=""):
        """Evaluate a single test set"""
        # Prepare evaluation data

        decoded_data = [
            self.tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
            for sample in data
        ]

        # Parse the decoded text to get target and nums
        test_prompts = []
        for text in decoded_data:
            # Extract target and nums from the decoded text
            # Format is expected to be "S target [ num1 num2 ... ] ,"
            parts = text.split("[")
            target = parts[0].split("S")[1].strip()
            nums = parts[1].split("]")[0].strip().split()

            # Reconstruct the prompt
            prompt = self.tokenizer.bos_token + f"S {target} [ {' '.join(nums)} ] ,"
            test_prompts.append(prompt)

        # Get predictions
        predictions = self.eval_ll(
            self.hf_model,
            self.tokenizer,
            test_prompts,
            batch_size=self.batch_size,
            context_len=self.config.model.block_size,
            temperature=0.0,
            n=1,
        )

        # Calculate metrics
        pred_ratings = []
        true_rating = []
        pred_reasons = []

        for i in range(len(predictions)):
            rating, reason = metric_fn(
                predictions[i]
                .split(self.tokenizer.bos_token)[1]
                .split(self.tokenizer.eos_token)[0],
                mode="sft",
            )
            tr, _ = metric_fn(f"{decoded_data[i]}", mode="sft")
            print(tr, reason)
            pred_ratings.append(rating)
            true_rating.append(tr)
            pred_reasons.append(reason)

        pred_ratings = np.array(pred_ratings)
        metrics = {
            f"{prefix}average_rating": float(np.mean(pred_ratings)),
            f"{prefix}average_true_rating": float(np.mean(true_rating)),
            f"{prefix}accuracy": float(np.mean([r > 0 for r in pred_ratings])),
            f"{prefix}true_accuracy": float(np.mean([r > 0 for r in true_rating])),
        }

        return metrics, predictions, pred_ratings, pred_reasons, test_prompts

    def evaluate(self, test_all_data, test_first_data, test_last_data):
        """Evaluate all test sets using the data from the dataloaders"""
        try:
            self.hf_model = convert_litgpt_to_hf(self.config)
            self.hf_model.cuda()
            self.hf_model.eval()

            # Evaluate main test set
            print("\nEvaluating test_all set...")
            main_metrics, predictions, pred_ratings, pred_reasons, test_prompts = (
                self.evaluate_dataset(test_all_data, prefix="countdown_eval/")
            )

            # Save results for test_all
            eval_dir = os.path.join(self.config.eval.results_dir, f"step_{self.step}")
            os.makedirs(eval_dir, exist_ok=True)
            results_file = os.path.join(eval_dir, f"results_{self.num_examples}.json")
            with open(results_file, "w") as f:
                json.dump(
                    {
                        "trajectories": predictions,
                        "ratings": pred_ratings.tolist(),
                        "reasons": pred_reasons,
                        "test_prompts": test_prompts,
                    },
                    f,
                    indent=4,
                )

            # Evaluate test_first set
            print("\nEvaluating test_first set...")
            first_metrics, _, _, _, _ = self.evaluate_dataset(
                test_first_data, prefix="countdown_eval_first/"
            )

            # Evaluate test_last set
            print("\nEvaluating test_last set...")
            last_metrics, _, _, _, _ = self.evaluate_dataset(
                test_last_data, prefix="countdown_eval_last/"
            )

            # Combine all metrics
            all_metrics = {**main_metrics, **first_metrics, **last_metrics}

            # Save to CSV (for test_all)
            if not any(self.results_df["step"] == self.global_step):
                new_row = pd.DataFrame(
                    [
                        {
                            "step": self.global_step,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "average_rating": main_metrics[
                                "countdown_eval/average_rating"
                            ],
                            "average_true_rating": main_metrics[
                                "countdown_eval/average_true_rating"
                            ],
                            "accuracy": main_metrics["countdown_eval/accuracy"],
                            "true_accuracy": main_metrics[
                                "countdown_eval/true_accuracy"
                            ],
                            "predictions": json.dumps(predictions),
                        }
                    ]
                )
                self.results_df = pd.concat(
                    [self.results_df, new_row], ignore_index=True
                )
                self.results_df.to_csv(self.csv_path, index=False)

            return all_metrics

        except Exception as e:
            print(f"Error during countdown evaluation: {e}")
            raise e

        finally:
            del self.hf_model
            torch.cuda.empty_cache()
