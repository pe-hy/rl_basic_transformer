from argparse import ArgumentParser
from urllib.request import urlopen

import lightning as L
import torch
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger

#from lightning_gpt import callbacks, data
import functools
import warnings
from typing import Any, Optional, Tuple

import torch.optim
from lightning import LightningModule
import models

class NanoGPT(LightningModule):

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: Optional[int] = None,
        n_head: Optional[int] = None,
        n_embd: Optional[int] = None,
        dropout: float = 0.0,
        weight_decay: float = 0.1,
        learning_rate: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.95)
    ):
        super().__init__()
        self.betas = betas
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.config = models.GPTConfig(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd, dropout=dropout)
        self.gpt = models.GPT(self.config)

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
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        idx, targets = batch
        _, loss1 = self(idx, targets)
        self.log("validation_loss", loss1,prog_bar=True)
        return loss1

def get_random_bool(num_samples,input_length):
    # create a dataset of num_samples samples with strings of bools of length input_length as input x and random bool as output y
    x = torch.randint(0, 2, (num_samples, input_length))
    y = torch.randint(0, 2, (num_samples, 1))
    return x,y



def main(args):
    # create torch dataset for train and test
    train_data= get_random_bool(100, args.block_size) 
    test_data = get_random_bool(10, args.block_size)
    # convert train data to torch dataset

    train_dataset = torch.utils.data.TensorDataset(train_data[0], train_data[1])
    test_dataset = torch.utils.data.TensorDataset(test_data[0], test_data[1])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,shuffle=False)

    callback_list = []
    model = NanoGPT(vocab_size=2,block_size=args.block_size,n_layer=args.n_layer,n_head=args.n_head,n_embd=args.n_embd)

    wandb_logger = WandbLogger(project=args.project_name, name=args.data_folder, save_dir=args.data_folder)
    trainer = L.Trainer.from_argparse_args(
        args,
        max_epochs=args.num_epochs,
        gradient_clip_val=1.0,
        callbacks=callback_list,
        accelerator="auto",
        logger=wandb_logger,
        #devices=args.devices,
        precision=16,
    )

    trainer.fit(model, train_loader, test_loader)


if __name__ == "__main__":
    L.seed_everything(42)

    parser = ArgumentParser()
    parser = L.Trainer.add_argparse_args(parser)
    parser.add_argument("--project_name", default='cc', type=str)
    parser.add_argument("--data_folder", default='data',type=str)
    parser.add_argument("--n_layer", default= 8,type=int)
    parser.add_argument("--n_head", default=4,type=int)
    parser.add_argument("--n_embd", default=64,type=int)
    parser.add_argument("--n_hidden", default=64,type=int)
    parser.add_argument("--tied", type=bool)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--wdecay", default=1.2e-6, type=float)
    parser.add_argument("--block_size", default=8, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    args = parser.parse_args()

    main(args)