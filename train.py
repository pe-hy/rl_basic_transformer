import argparse
import pickle
import logging
import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.loggers import WandbLogger
from typing import Optional, Tuple
import torch.optim
from lightning import LightningModule
import modular_models_2_att as modular_models
from model import GPTConfig, GPT
from types import SimpleNamespace
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import numpy as np

log = logging.getLogger(__name__)

class NanoGPT(LightningModule):
    def __init__(
        self,
        model_config: SimpleNamespace,
        dropout: float = 0.0,
        weight_decay: float = 0.1,
        learning_rate: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.95),
    ):
        super().__init__()
        self.betas = betas
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=['model_config'])
        self.gpt = modular_models.ModularGPT(model_config)
        #self.gpt = GPT(model_config)

        self.validation_step_outputs = []

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.gpt(idx, targets)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.gpt.configure_optimizers(
            weight_decay=self.weight_decay,
            learning_rate=self.learning_rate,
            betas=self.betas
        )

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        idx, targets = batch
        _, loss = self(idx, targets)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        idx, targets = batch
        logits, loss = self(idx, targets)
        accuracy = self.calculate_accuracy(logits, targets)
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        
        self.validation_step_outputs.append(accuracy)
        return {'val_loss': loss, 'val_accuracy': accuracy}

    def on_validation_epoch_end(self):
        avg_accuracy = torch.stack(self.validation_step_outputs).mean()
        self.log('epoch_val_accuracy', avg_accuracy, prog_bar=True)
        print(f"Validation Epoch End: Accuracy = {avg_accuracy.item():.4f}")
        self.validation_step_outputs.clear()

    def calculate_accuracy(self, logits, targets):
        predictions = torch.argmax(logits, dim=-1)
        targets = targets[:, 0] # prev_enabling, enabling, subgoal, prev_start, prev_subgoal
        # preds_list = []
        # targs_list = []
        # for i in range(50):
        #     preds = predictions[i]
        #     targs = targets[i]
        #     # get index of 199 in targs
        #     index_of_199 = (targs == 199).nonzero(as_tuple=True)[0]
        #     preds_list.append(preds[1:index_of_199])
        #     targs_list.append(targs[1:index_of_199])
        # predictions = torch.cat(preds_list)
        # targets = torch.cat(targs_list)
        return (predictions == targets).float().mean()

def get_data(data, input_length, flag):
    dataset = data["test"] if flag else data["train"]
    num_samples = len(dataset)
    x = torch.zeros((num_samples, input_length), dtype=torch.long)
    y = torch.zeros((num_samples, 1), dtype=torch.long)
    for i, sample in enumerate(dataset):
        x[i] = torch.tensor(sample['input'])
        y[i] = torch.tensor([sample["target"]])
    return x, y

class SequenceDataset_lol(Dataset):
    '''
    Dataset and DataLoader for sequence data.
    Made specifically for autoregressive next token prediction training
    Data is integer-type for tokenizer
    '''

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

def collate_fn_pad(batch):
    x, y = zip(*batch)
    # Pad sequences to the maximum length in the batch
    x_padded = pad_sequence(x, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y, batch_first=True, padding_value=0)
    return x_padded, y_padded



def main(args):

    args.init_bottleneck_by_last = args.init_bottleneck_by_last == 'True'
    
    with open(args.datapath, "rb") as f:
        data = pickle.load(f)

    num_tokens_in_str = len(data['train'][0]['input'])
    #train_data = get_data(data, num_tokens_in_str, False) 
    #test_data = get_data(data, num_tokens_in_str, True)
    vocab_size = len(data['vocab'])+100 #len(data['vocab']) #+ num_tokens_in_str - 1
    block_size = num_tokens_in_str #len(data['train'][0]['input'])

    #train_dataset = TensorDataset(train_data[0], train_data[1])
    #test_dataset = TensorDataset(test_data[0], test_data[1])
    #train_path ='/home/jan/projects/HARVARD/mikail/graph_transformer/data/tokens_path_train.npy'
    #test_path = '/home/jan/projects/HARVARD/mikail/graph_transformer/data/tokens_path_eval.npy'
    train_dataset = SequenceDataset_lol(data['train'], add_one_token=True)
    test_dataset = SequenceDataset_lol(data['test'], add_one_token=True,test=True)
    #train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=False, collate_fn=collate_fn_pad)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False, collate_fn=collate_fn_pad)
    # Create a SimpleNamespace object for the model configuration
    model_config = SimpleNamespace(
        divider = args.divider,
        vocab_size=vocab_size,
        block_size=block_size,
        n_iters=args.n_iters,
        n_head=args.n_head,
        n_embd=args.n_embd,
        n_hidden=args.n_hidden,
        dropout=args.dropout,
        bias=args.bias,
        init_bottleneck_by_last=args.init_bottleneck_by_last
    )


    model = NanoGPT(
        model_config=model_config,
        dropout=args.dropout,
        weight_decay=args.wdecay,
        learning_rate=args.learning_rate,
    )


    run_name = f"ib_{args.init_bottleneck_by_last}_ni_{args.n_iters}_divider_{args.divider}_vs_{args.vocab_size}_ne_{args.n_embd}_nh_{args.n_hidden}_bs_{args.batch_size}_bls_{args.block_size}"
    #wandb_logger = WandbLogger(project="rl_transformer_divider_cube", name=run_name, save_dir="wandb_saves")

    trainer = L.Trainer.from_argparse_args(
        args,
        max_epochs=args.num_epochs,
        gradient_clip_val=1.0,
        accelerator="cuda",
        #logger=wandb_logger,
        #precision=16,
    )

    trainer.fit(model, train_loader, test_loader)
    # Save the model
    model_save_path = f"/home/jan/projects/HARVARD/rl_basic_transformer/Cube/saved_models/model.pt"
    torch.save(model.state_dict(), model_save_path)
    # same the config
    with open(f"/home/jan/projects/HARVARD/rl_basic_transformer/Cube/saved_models/model_config.pkl", "wb") as f:
        pickle.dump(model_config, f)
    log.info(f"Model saved to {model_save_path}")

    accuracy = trainer.validate(model, test_loader)[0]['val_accuracy_epoch']
    log.info(f"Model accuracy: {accuracy}")
    return accuracy


if __name__ == "__main__":
    L.seed_everything(42)

    parser = argparse.ArgumentParser()
    parser = L.Trainer.add_argparse_args(parser)

    # Changing
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=200)
    parser.add_argument("--block_size", type=int, default=17)
    parser.add_argument("--n_iters", type=int, default=3)
    parser.add_argument("--n_embd", type=int, default=96)
    parser.add_argument("--n_hidden", type=int, default=16)
    parser.add_argument("--init_bottleneck_by_last", type=str, default='False')
    parser.add_argument("--divider", type=int, default=1)
    parser.add_argument("--datapath", type=str, default="/home/jan/projects/HARVARD/rl_basic_transformer/data.pkl")

    # Unchanging
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bias", type=bool, default=False)
    parser.add_argument("--learning_rate", default=6e-4, type=float)
    parser.add_argument("--wdecay", default=0.1, type=float)
    # Old
    # parser.add_argument("--project_name", default='cc', type=str)  # unused
    # parser.add_argument("--data_folder", default='data', type=str)  # unused
    # parser.add_argument("--n_layer", default=2, type=int)  # unused
    # parser.add_argument("--tied", type=bool)  # unused
    args = parser.parse_args()
    main(args)