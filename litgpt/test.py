import torch
from transformers import AutoModel

from litgpt import LLM

model = LLM.load("/mnt/raid/data/Hyner_Petr/rl/sos_branch/rl_basic_transformer/litgpt/trained_model")
print(model.preprocessor.tokenizer)
#print(model)