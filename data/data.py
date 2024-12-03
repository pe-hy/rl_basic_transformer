import pickle
import torch
from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule

from torch.nn.utils.rnn import pad_sequence

from omegaconf import DictConfig, OmegaConf

class Datamodule(LightningDataModule):
    def __init__(self, datasets, batch_size, num_workers):
        super(Datamodule, self).__init__()
        self.datasets = datasets
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = self.dataset['train']
        self.val_dataset = self.dataset['val']
        self.test_dataset = self.dataset['test']

    def collate_fn_pad(self,batch):
        x, y = zip(*batch)
        # Pad sequences to the maximum length in the batch
        x_padded = pad_sequence(x, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)
        return x_padded, y_padded

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=self.num_workers,
                                drop_last=False,
                                collate_fn=self.collate_fn_pad)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=False,
                          collate_fn=self.collate_fn_pad)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=False,
                          collate_fn=self.collate_fn_pad)




class SequenceDataset(Dataset):

    def __init__(self, filepath, add_one_token=True,test=False):
        #data = np.load(filepath, allow_pickle=True)
        data = filepath
        target_key = 'key'
        self.inputs = [list(x['input']) for x in data] #if len(x['input']) == 14]  # Loading the sequences as lists
        self.enabling = [x['enabling'] for x in data]
        self.subgoal = [x['subgoal'] for x in data]
        self.prev_enabling = [x['prev_enabling'] for x in data]
        self.prev_subgoal = [x['prev_subgoal'] for x in data]
        self.prev_start = [x['prev_start'] for x in data]
        self.prev_prev_enabling = [x['prev_prev_enabling'] for x in data]
        if test:
            max_size = max(5000,len(self.inputs))
            self.inputs = self.inputs[:max_size]
            self.enabling = self.enabling[:max_size]
            self.subgoal = self.subgoal[:max_size]
            self.prev_enabling = self.prev_enabling[:max_size]
            self.prev_subgoal = self.prev_subgoal[:max_size]
            self.prev_start = self.prev_start[:max_size]
            self.prev_prev_enabling = self.prev_prev_enabling[:max_size]
        self.add_one_token = add_one_token

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # y is 1-index shifted version of x. Everything should be integer for tokenizer.
        start = 0
        x = self.inputs[idx][start:]
        y = [self.prev_prev_enabling[idx],self.prev_enabling[idx],self.enabling[idx],self.subgoal[idx],self.prev_start[idx],self.prev_subgoal[idx]]  
        return torch.tensor(x, dtype=torch.int64), torch.tensor(y, dtype=torch.int64)


def get_data(data: DictConfig):
    with open(data.datapath, "rb") as f:
        data = pickle.load(f)
    train_dataset = SequenceDataset(data['train'], add_one_token=True)
    test_dataset = SequenceDataset(data['test'], add_one_token=True,test=True)
    return {'train':train_dataset, 
            'test':test_dataset, 
            'val':test_dataset}