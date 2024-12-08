import os
import json
import random
import argparse

import tqdm
from litgpt import LLM
from litgpt.generate.base import generate
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, GPTNeoForCausalLM, AutoModel
from datasets import load_dataset, DatasetDict, Dataset

from countdown_utils import *
from countdown_bfs import bfs
from countdown_dfs import dfs

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=4)
parser.add_argument("--ckpt", type=str, help="path to checkpoint")
parser.add_argument("-n", "--num",type=int, default=10)
parser.add_argument("-o", "--offset",type=int, default=0)
parser.add_argument("--data_dir", type=str, default="data/")
parser.add_argument("-d", "--data",type=str, default="val_b3_t100_n100000_random.json")
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--ctx", type=int, default=4096)
parser.add_argument("--gens", type=int, default=1)

@torch.inference_mode()
def generate_batch(
    model,
    idx: torch.Tensor,
    max_returned_tokens: int,
    *,
    temperature: float = 1.0,
    top_k = None,
    eos_id = None,
) -> torch.Tensor:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        idx: Tensor of shape (B, T) with indices of the prompt sequence.
        max_returned_tokens: The maximum number of tokens to return (given plus generated).
        temperature: Scales the predicted logits by 1 / temperature.
        top_k: If specified, only sample among the tokens with the k highest probabilities.
        eos_id: If specified, stop generating any more token once the <eos> token is triggered.
    """
    device, dtype = idx.device, idx.dtype
    B = idx.size(0)
    T = idx.size(1)
    V = model.config.padded_vocab_size
    seq_lengths = (idx == eos_id).to(torch.int16).argmax(dim=1, keepdim=True)
    seq_lengths = torch.where(seq_lengths == 0, torch.full_like(seq_lengths, T), seq_lengths)
    seq_lengths -= 1 # this way, seq_lengths presents the index for inserting tokens for each samples
    is_complete = torch.zeros(B, device=seq_lengths.device)

    # assert max_returned_tokens > T
    # if model.max_seq_length < max_returned_tokens - 1:
    #     # rolling the kv cache based on the `input_pos` value would be necessary. However, doing so would introduce a
    #     # data dependency on the `input_pos` tensor and impact model compilation. Since this setting is uncommon, we do
    #     # not support it to avoid negatively impacting the overall speed
    #     raise NotImplementedError(f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}")

    device, dtype = idx.device, idx.dtype
    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(B, max_returned_tokens, dtype=dtype, device=device)
    empty[:, :T] = idx
    idx = empty

    # generate up to a fixed number of tokens
    for step in range(max_returned_tokens  - T):
        # forward
        print("-"*30)
        print(idx.shape)
        full_logits = model(idx) # B * T * V
        logits = torch.gather(full_logits, 1, seq_lengths.unsqueeze(2).tile(1, 1, V)).squeeze(1) # B * V
        logits /= temperature + 1e-9

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(logits < v[:, -1].unsqueeze(1), -float("Inf"), logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1, replacement=True).to(dtype=dtype)

        # sactter the generated tokens (idx_next) to appropriate indices
        seq_lengths += 1
        torch.scatter(idx, 1, seq_lengths, idx_next)

        # if <eos> token is generated, return the output (stop generation)
        is_complete = torch.logical_or(is_complete, idx_next.squeeze(1) == eos_id)
        if torch.all(is_complete):
            break

    return idx

def eval_ll(model, tokenizer, data, batch_size=1, context_len=4104, temperature=0.0, n=1):
    """
    Evaluate the model on the data using a sliding window so that the context length is not exceeded
    """
    output_texts_concat = []
    for b in tqdm.trange(0, len(data), batch_size):
        batch = data[b:min(b+batch_size, len(data))]
        output_texts = ["" for _ in range(len(batch))]
        tokenizer.padding_side = "left"
        inputs = tokenizer.encode(batch[0], return_tensors="pt").to("cpu")
        inputs = inputs.unsqueeze(0)
        # print(batch[0])
        # print(model._text_to_token_ids(batch[0]))
        # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):

        if n == 1:
            if temperature == 0.0:
                # outputs = model.generate(input_ids=inputs, pad_token_id=tokenizer.eos_token_id, attention_mask=torch.ones_like(inputs), max_length=context_len, num_beams=1, do_sample=False)
                # outputs = generate_batch(model, idx=inputs, eos_id=tokenizer.eos_token_id, max_returned_tokens=context_len, temperature=temperature)
                model.model.max_seq_length = context_len
                # enable the kv cache
                model.model.set_kv_cache(batch_size=1)
                # model.cuda()
                outputs = generate(model=model.model, prompt=inputs, max_returned_tokens=context_len, temperature=temperature, eos_id=tokenizer.eos_token_id, include_prompt=True)
                for block in model.model.transformer.h:
                    block.attn.kv_cache.reset_parameters()
            else:
                outputs = generate(model=model.model, prompt=inputs, max_returned_tokens=context_len, temperature=temperature, eos_id=tokenizer.eos_token_id, include_prompt=True)
            # split output vector into first N tokens and the rest
            output_tokens = outputs
            output_text = tokenizer.decode(output_tokens, skip_special_tokens=False)
            tokenizer.padding_side = "left"
            output_texts = [ot + ot_now for ot, ot_now in zip(output_texts, [output_text])]
            # print token lens of tokenized outputs
            print([len(tokenizer(ot)['input_ids']) for ot in output_texts])
            output_texts_concat += output_texts
        else:
            assert temperature > 0.0, "Temperature must be greater than 0 for sampling"
            all_outputs = []
            all_ratings = []
            for i in range(n):
                output_text = generate_batch(model, idx=inputs, eos_id=tokenizer.eos_token_id, max_returned_tokens=context_len, temperature=temperature)
                # output_tokens = outputs
                # output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=False)
                # tokenizer.padding_side = "left"
                # get rating for each output
                ratings = [metric_fn(ot.split(tokenizer.bos_token)[1], mode="sft")[0] for ot in output_text]
                all_ratings.append(ratings)
                all_outputs.append(output_text)
            # only keep the output with the highest rating for each input
            all_ratings = np.array(all_ratings)

            print(all_ratings)
            print(f"average rating", np.mean(all_ratings))
            # all ratings is n x batch_size
            max_ratings = np.argmax(all_ratings, axis=0)
            max_rating_vals = np.max(all_ratings, axis=0)
            print(f"max ratings", np.mean(max_rating_vals))
            # max ratings is batch_size, output_texts is n x batch_size
            output_texts = [all_outputs[max_r][i] for i, max_r in enumerate(max_ratings)]
            output_texts_concat += output_texts
    return output_texts_concat 

args = parser.parse_args()
torch.manual_seed(args.seed)
# state_dict = torch.load("/mnt/raid/data/Hyner_Petr/rl/sos_branch/rl_basic_transformer/litgpt/hf_trained_model/model.pth")
# model = AutoModel.from_pretrained(args.ckpt, local_files_only=True, state_dict=state_dict)
# model = GPTNeoForCausalLM.from_pretrained(args.ckpt, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2')
model = LLM.load("/mnt/raid/data/Hyner_Petr/rl/sos_branch/rl_basic_transformer/litgpt/trained_model")
model.eval()
model.cpu()
#model.cuda()

tokenizer = AutoTokenizer.from_pretrained(args.ckpt, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token

data_file = os.path.join(args.data_dir, args.data)

with open(data_file, "r") as json_file:
    data = json.load(json_file)

predictions = []
pred_ratings = []
pred_reasons = []
tokenizer.padding_side = "left"
test_prompts = [tokenizer.bos_token + f" S {sample['target']} [ {' '.join(map(str,sample['nums']))} ] ," 
                          for sample in data[:100]] # TODO
len_nums = [len(sample['nums']) for sample in data[args.offset:args.num]]
data_4 = [d for d, l in zip(test_prompts, len_nums) if l == 4]
predictions = eval_ll(model, tokenizer, data_4, batch_size=1, context_len=4088, temperature=args.temperature, n=args.gens)

len_pred_nums = [4 for _ in predictions]

# rate outputs
true_rating = []
for i in range(len(predictions)):
    rating, reason = metric_fn(predictions[i].split(tokenizer.bos_token)[1], mode="sft")
    tr, _ = metric_fn(f"{data[i]['search_path']}", mode="sft")
    pred_ratings.append(rating)
    true_rating.append(tr)
    pred_reasons.append(reason)

# get max rating for each sample with its index
pred_ratings = np.array(pred_ratings)

# print results
print("Results Summary:")
print(f"Average rating: {np.mean(pred_ratings)}")
print(f"Average true rating: {np.mean(true_rating)}")
print(f"Accuracy: {np.mean([r > 0 for r in pred_ratings])}")
print(f"True Accuracy: {np.mean([r > 0 for r in true_rating])}")

ckpt_dir = os.path.dirname(args.ckpt)
# save results
results_file = os.path.join(ckpt_dir, f"results_{args.data.replace('/','_')}_{args.num}_{args.offset}")
with open(results_file, "w") as f:
    json.dump({"trajectories": predictions, "ratings": pred_ratings.tolist(), "reasons": pred_reasons}, f, indent=4)