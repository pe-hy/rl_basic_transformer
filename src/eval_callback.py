from transformers import TrainerCallback
import os
import json
import torch
import numpy as np
import wandb
import pandas as pd
from datetime import datetime
from countdown_utils import *
from tqdm import trange

class EvalCallback(TrainerCallback):
    def __init__(self, data_dir, eval_data, tokenizer, num_examples=10, batch_size=64):
        self.data_dir = data_dir
        self.eval_data = eval_data
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.last_eval_step = None
        
        # Load evaluation data once
        data_file = os.path.join(self.data_dir, self.eval_data)
        with open(data_file, "r") as json_file:
            self.data = json.load(json_file)
        
        # Create results directory if it doesn't exist
        self.results_dir = os.path.join(data_dir, "eval_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize results DataFrame
        self.csv_path = os.path.join(self.results_dir, "eval_results.csv")
        if os.path.exists(self.csv_path):
            self.results_df = pd.read_csv(self.csv_path)
        else:
            self.results_df = pd.DataFrame(columns=[
                'step', 
                'timestamp',
                'average_rating', 
                'average_true_rating',
                'accuracy',
                'true_accuracy'
            ])

    def eval_ll(self, model, tokenizer, data, batch_size=128, context_len=4096, temperature=0.0, n=1):
        """
        Evaluate the model on the data using a sliding window so that the context length is not exceeded
        """
        output_texts_concat = []
        for b in trange(0, len(data), batch_size):
            batch = data[b:min(b+batch_size, len(data))]
            output_texts = ["" for _ in range(len(batch))]
            tokenizer.padding_side = "left"
            inputs = tokenizer(batch, return_tensors="pt", padding=True).to("cuda")
            inputs = inputs['input_ids']

            if n == 1:
                outputs = model.generate(
                    input_ids=inputs, 
                    pad_token_id=tokenizer.eos_token_id, 
                    attention_mask=torch.ones_like(inputs), 
                    max_length=context_len, 
                    num_beams=1, 
                    do_sample=False
                )
                output_tokens = outputs
                output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=False)
                tokenizer.padding_side = "left"
                output_texts = [ot + ot_now for ot, ot_now in zip(output_texts, output_text)]
                output_texts_concat += output_texts

        return output_texts_concat 

    def on_train_begin(self, args, state, control, model=None, tokenizer=None, **kwargs):
        pass
            
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        # Skip if we've already evaluated at this step
        if self.last_eval_step == state.global_step:
            return
            
        if self.tokenizer is None:
            print("Warning: Tokenizer not initialized, skipping evaluation")
            return
            
        print(f"\nRunning custom countdown evaluation at step {state.global_step}")
            
        # Ensure model is in eval mode
        was_training = model.training
        model.eval()
        
        try:
            # Prepare evaluation data # S 17 [ 20 18 26 11 ] , TODO
            test_prompts = [self.tokenizer.bos_token + f"S {sample['target']} [ {' '.join(map(str,sample['nums']))} ] ," 
                          for sample in self.data[:self.num_examples]]
            # test_prompts = [self.tokenizer.bos_token + f"Current State: {sample['target']}:{sample['nums']}, Operations: []" 
            #               for sample in self.data[:self.num_examples]]
            len_nums = [len(sample['nums']) for sample in self.data[:self.num_examples]]
            data_4 = [d for d, l in zip(test_prompts, len_nums) if l == 4]
            
            # Get predictions using eval_ll
            predictions = self.eval_ll(
                model, 
                self.tokenizer, 
                data_4, 
                batch_size=self.batch_size, 
                context_len=4096, 
                temperature=0.0, 
                n=1
            )

            # Rate outputs
            pred_ratings = []
            true_rating = []
            pred_reasons = []
            
            for i in range(len(predictions)):
                rating, reason = metric_fn(predictions[i].split(self.tokenizer.bos_token)[1], mode="sft")
                tr, _ = metric_fn(f"{self.data[i]['search_path']}", mode="sft")
                pred_ratings.append(rating)
                true_rating.append(tr)
                pred_reasons.append(reason)
            
            # Calculate metrics
            pred_ratings = np.array(pred_ratings)
            avg_rating = float(np.mean(pred_ratings))
            avg_true_rating = float(np.mean(true_rating))
            accuracy = float(np.mean([r > 0 for r in pred_ratings]))
            true_accuracy = float(np.mean([r > 0 for r in true_rating]))
            
            # Save detailed results
            eval_dir = os.path.join(self.results_dir, f"step_{state.global_step}")
            os.makedirs(eval_dir, exist_ok=True)
            
            results_file = os.path.join(eval_dir, f"results_{self.eval_data.replace('/','_')}_{self.num_examples}_0")
            with open(results_file, "w") as f:
                json.dump({
                    "trajectories": predictions,
                    "ratings": pred_ratings.tolist(),
                    "reasons": pred_reasons
                }, f, indent=4)
            
            # Log to wandb if available
            if wandb.run is not None:
                metrics = {
                    "countdown_eval/average_rating": avg_rating,
                    "countdown_eval/average_true_rating": avg_true_rating,
                    "countdown_eval/accuracy": accuracy,
                    "countdown_eval/true_accuracy": true_accuracy,
                }
                wandb.log(metrics, step=state.global_step)
                print("Successfully logged countdown evaluation metrics to wandb")
            
            # Save to CSV only if we haven't already saved for this step
            if not any(self.results_df['step'] == state.global_step):
                new_row = pd.DataFrame([{
                    'step': state.global_step,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'average_rating': avg_rating,
                    'average_true_rating': avg_true_rating,
                    'accuracy': accuracy,
                    'true_accuracy': true_accuracy
                }])
                
                self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)
                self.results_df.to_csv(self.csv_path, index=False)
            
            # Print results summary
            print("\nResults Summary:")
            print(f"Average rating: {avg_rating}")
            print(f"Average true rating: {avg_true_rating}")
            print(f"Accuracy: {accuracy}")
            print(f"True Accuracy: {true_accuracy}")
            
            # Update last evaluated step
            self.last_eval_step = state.global_step
                
        except Exception as e:
            print(f"Error during countdown evaluation: {e}")
            raise e
            
        finally:
            # Restore model's training state
            if was_training:
                model.train()