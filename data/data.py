import pickle
import torch
from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedTokenizerFast
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_tokenizer(data: DictConfig):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=(data.tokenizer_path)
    )
    tokenizer.eos_token = "[EOS]"
    tokenizer.unk_token = "[UNK]"
    tokenizer.pad_token = "[PAD]"
    tokenizer.mask_token = "[MASK]"
    tokenizer.bos_token = "[BOS]"
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

class SequenceDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        tokens = item['tokens']
        pos_embeddings = torch.tensor(item['pos_embeddings'], dtype=torch.long)
        masks = torch.tensor(item['masks'], dtype=torch.float)
        labels = torch.tensor(item['labels'], dtype=torch.float)

        return {
            'tokens': tokens,
            'pos_embeddings': pos_embeddings,
            'masks': masks,
            'labels': labels
        }

class Datamodule(LightningDataModule):
    def __init__(self, datasets, batch_size, num_workers):
        super(Datamodule, self).__init__()
        self.datasets = datasets
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = self.datasets['train']
        self.val_dataset = self.datasets['val']
        self.test_dataset = self.datasets['test']

    def collate_fn_pad(self, batch):
        # Extract each component from the batch
        tokens = [item['tokens'] for item in batch]
        pos_embeddings = [item['pos_embeddings'] for item in batch]
        masks = [item['masks'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        # print(tokens)
        # Convert tokens to tensors if they aren't already
        if not isinstance(tokens[0], torch.Tensor):
            tokens = [torch.tensor(t, dtype=torch.long) for t in tokens]
        if not isinstance(pos_embeddings[0], torch.Tensor):
            pos_embeddings = [torch.tensor(p, dtype=torch.long) for p in pos_embeddings]
        if not isinstance(masks[0], torch.Tensor):
            masks = [torch.tensor(m, dtype=torch.float) for m in masks]
        if not isinstance(labels[0], torch.Tensor):
            labels = [torch.tensor(l, dtype=torch.float) for l in labels]
        
        return {
            'tokens': torch.tensor(np.asarray(tokens)),
            'pos_embeddings': torch.tensor(np.asarray(pos_embeddings)),
            'masks': torch.tensor(np.asarray(masks)),
            'labels': torch.tensor(np.asarray(labels))
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.collate_fn_pad
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.collate_fn_pad
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=self.collate_fn_pad
        )

def get_data(data: DictConfig):

    with open(data.datapath_train, "rb") as f:
        train = pickle.load(f)
    with open(data.datapath_val, "rb") as f:
        test = pickle.load(f)

    tokenizer = get_tokenizer(data)
    
    processed_train = {}
    for idx in train:
        processed_train[idx] = train[idx].copy()
        toks = []

        for i in train[idx]["tokens"]:
            toks.append(tokenizer.encode(i)[0])

        processed_train[idx]['tokens'] = toks
    
    processed_test = {}
    for idx in test:
        processed_test[idx] = test[idx].copy() 

        toks = []
        for i in test[idx]["tokens"]:
            toks.append(tokenizer.encode(i)[0])
        
        processed_test[idx]['tokens'] = toks

    train_dataset = SequenceDataset(processed_train)
    test_dataset = SequenceDataset(processed_test)
    
    return {
        'train': train_dataset,
        'test': test_dataset,
        'val': test_dataset
    }