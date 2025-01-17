import torch
import random
import numpy as np

# 4 tensory

literals = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
clauses = ["1", "2", "3", "4", "5"]

ret = dict()

for i in range(0, 20):
    rand_lit = random.sample(
    literals,
    k=3,
)
    rand_clause = random.sample(clauses, k=4)
    sample = rand_lit + rand_clause

    n = len(sample)
    k = 3
    l = len(literals)
    pos_embeddings = np.random.choice((range(1, 10)), size=(k, n))

    masks = np.random.choice([0, 1], size=(k, n), p=[0.5, 0.5]) # jedničky tam kde jsou literaly, 0 jinde
    labels = np.random.choice([0, 1], size=(l, n), p=[0.5, 0.5])
    ret[i] = {
        "tokens": sample,
        "pos_embeddings": pos_embeddings,
        "masks": masks,
        "labels": labels, # počet řádku - počet literálů, sloupců je počet tokenů v sekvenci
    }

import pickle
with open("test.pkl", "wb") as f:
    pickle.dump(ret, f)

with open("train.pkl", "wb") as f:
    pickle.dump(ret, f)
