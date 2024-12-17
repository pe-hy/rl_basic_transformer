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


class EvalCallback(Callback):
    def __init__(
        self,
        data_dir,
        eval_data,
        tokenizer,
        num_examples=128,
        batch_size=64,
        save_path=None,
        config=None,
        eval_interval=1000,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.eval_data = eval_data
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.last_eval_step = -1  # Initialize to -1
        self.config = config
        self.hf_model = None
        self.eval_interval = eval_interval
        self.save_path = save_path

        # Load evaluation data once
        data_file = os.path.join(self.data_dir, self.eval_data)
        with open(data_file, "r") as json_file:
            self.data = json.load(json_file)

        # Create results directory if it doesn't exist
        self.results_dir = self.config.eval.results_dir
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
        """
        Evaluate the model on the data using a sliding window so that the context length is not exceeded
        """
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

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Only run evaluation at specified intervals and if we haven't evaluated at this step
        if (
            trainer.global_step % self.eval_interval == 0
            and trainer.global_step > self.last_eval_step
            and trainer.is_global_zero
        ):
            print(f"Saving model before evaluation...")
            pl_module.llm.model.to(pl_module.llm.preprocessor.device)
            pl_module.llm.save(self.save_path)
            self.run_evaluation(trainer, pl_module)

    def run_evaluation(self, trainer, pl_module):
        print(f"\nRunning custom countdown evaluation at step {trainer.global_step}")

        try:
            self.hf_model = convert_litgpt_to_hf(self.config)
            self.hf_model.cuda()
            self.hf_model.eval()

            # Prepare evaluation data
            test_prompts = [
                self.tokenizer.bos_token
                + f"S {sample['target']} [ {' '.join(map(str,sample['nums']))} ] ,"
                for sample in self.data[: self.num_examples]
            ]
            len_nums = [
                len(sample["nums"]) for sample in self.data[: self.num_examples]
            ]
            data_4 = [d for d, l in zip(test_prompts, len_nums) if l == 4]

            # Get predictions
            predictions = self.eval_ll(
                self.hf_model,
                self.tokenizer,
                data_4,
                batch_size=self.batch_size,
                context_len=4096,
                temperature=0.0,
                n=1,
            )

            # Calculate metrics
            pred_ratings = []
            true_rating = []
            pred_reasons = []

            for i in range(len(predictions)):
                rating, reason = metric_fn(
                    predictions[i].split(self.tokenizer.bos_token)[1], mode="sft"
                )
                tr, _ = metric_fn(f"{self.data[i]['search_path']}", mode="sft")
                pred_ratings.append(rating)
                true_rating.append(tr)
                pred_reasons.append(reason)

            pred_ratings = np.array(pred_ratings)
            avg_rating = float(np.mean(pred_ratings))
            avg_true_rating = float(np.mean(true_rating))
            accuracy = float(np.mean([r > 0 for r in pred_ratings]))
            true_accuracy = float(np.mean([r > 0 for r in true_rating]))

            # Save detailed results
            eval_dir = os.path.join(
                self.config.eval.results_dir, f"step_{trainer.global_step}"
            )
            os.makedirs(eval_dir, exist_ok=True)

            results_file = os.path.join(
                eval_dir,
                f"results_{self.num_examples}_{self.eval_data.replace('/','_')}",
            )
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

            self.last_eval_step = trainer.global_step

            # Log using the trainer's logger instead of wandb directly
            metrics = {
                "countdown_eval/average_rating": avg_rating,
                "countdown_eval/average_true_rating": avg_true_rating,
                "countdown_eval/accuracy": accuracy,
                "countdown_eval/true_accuracy": true_accuracy,
            }

            # Use trainer's logger to log metrics
            for key, value in metrics.items():
                trainer.logger.log_metrics({key: value}, step=trainer.global_step)
            print("Successfully logged countdown evaluation metrics")

            # Save to CSV
            if not any(self.results_df["step"] == trainer.global_step):
                new_row = pd.DataFrame(
                    [
                        {
                            "step": trainer.global_step,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "average_rating": avg_rating,
                            "average_true_rating": avg_true_rating,
                            "accuracy": accuracy,
                            "true_accuracy": true_accuracy,
                            "predictions": json.dumps(predictions),
                        }
                    ]
                )

                self.results_df = pd.concat(
                    [self.results_df, new_row], ignore_index=True
                )
                self.results_df.to_csv(self.csv_path, index=False)

            # Print results summary
            print("\nResults Summary:")
            print(f"Average rating: {avg_rating}")
            print(f"Average true rating: {avg_true_rating}")
            print(f"Accuracy: {accuracy}")
            print(f"True Accuracy: {true_accuracy}")

        except Exception as e:
            print(f"Error during countdown evaluation: {e}")
            raise e

        finally:
            # Cleanup
            del self.hf_model
            torch.cuda.empty_cache()
