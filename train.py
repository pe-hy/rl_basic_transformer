from argparse import ArgumentParser
from urllib.request import urlopen

import pickle
import logging
import lightning as L
import torch
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from bunch_py3 import bunchify
#from lightning_gpt import callbacks, data
import functools
import warnings
from typing import Any, Optional, Tuple
from omegaconf import DictConfig, OmegaConf
import hydra
import torch.optim
from lightning import LightningModule
import models
import modular_models

log = logging.getLogger(__name__)

class NanoGPT(LightningModule):

    def __init__(
        self,
        cfg: DictConfig,
        dropout: float = 0.0,
        weight_decay: float = 0.1,
        learning_rate: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.95),
    ):
        super().__init__()
        self.betas = betas
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        #self.config = models.GPTConfig(vocab_size=vocab_size, block_size=block_size, n_iters=n_iters, n_head=n_head, n_embd=n_embd, dropout=dropout)
        #self.gpt = models.GPT(self.config)
        self.gpt = modular_models.ModularGPT(cfg.model)
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
    
    # def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
    #     idx, targets = batch
    #     _, loss1 = self(idx, targets)
    #     self.log("validation_loss", loss1,prog_bar=True)
    #     return loss1
    
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
        return (predictions == targets).float().mean()

def get_data(data, input_length, flag):
    if flag:
        dataset = data["test"]
    else:
        dataset = data["train"]
    
    num_samples = len(dataset)
    # Initialize x and y with the correct shapes
    x = torch.zeros((num_samples, input_length), dtype=torch.long)
    y = torch.zeros((num_samples, 1), dtype=torch.long)
    # Fill x and y with data
    for i, sample in enumerate(dataset):
        # print(input_length, len(sample["input"]))
        x[i] = torch.tensor(sample['input'])
        y[i] = torch.tensor([sample["positions_extended"]['final_goal']])
    return x, y

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
#     cfg = {
#     "data": {
#         "batch_size": 64,
#         "num_workers": 4,
#         "dataset_path": "/mnt/raid/data/Hyner_Petr/rl/rl_basic_transformer/data_cube.pkl"
#     },
#     "train": {
#         "learning_rate": 3.0e-4,
#         "num_epochs": 35,
#         "log_every": 100,
#         "lr": 5.0e-4,
#         "wdecay": 1.2e-6
#     },
#     "model": {
#         "vocab_size": 577,
#         "block_size": 70,
#         "n_iters": 2,
#         "n_head": 1,
#         "n_embd": 96,
#         "n_hidden": 16,
#         "dropout": 0.0,
#         "bias": True,
#         "init_bottleneck_by_last": True
#     }
# }   
#     cfg = bunchify(cfg)
    with open (cfg.data.dataset_path, "rb") as f:
        data = pickle.load(f)
    # create torch dataset for train and test
    train_data = get_data(data, 65, False) 
    test_data = get_data(data, 65, True)
    # convert train data to torch dataset

    train_dataset = torch.utils.data.TensorDataset(train_data[0], train_data[1])
    test_dataset = torch.utils.data.TensorDataset(test_data[0], test_data[1])

    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, shuffle=False)

    callback_list = []
    model = NanoGPT(cfg)

    #wandb_logger = WandbLogger(project=args.project_name, name=args.data_folder, save_dir=args.data_folder)
    trainer = L.Trainer(
        #args,
        max_epochs=cfg.train.num_epochs,
        gradient_clip_val=1.0,
        #callbacks=callback_list,
        accelerator="cuda",
        #logger=wandb_logger,
        #devices=args.devices,
        precision=16,
        #val_check_interval=1500,
        #checkpoint_callback=True,
        #default_root_dir="/home/p23131/rl/basic_transformer/model"
    )

    trainer.fit(model, train_loader, test_loader)
    # get the validation accuracy

    accuracy = trainer.validate(model, test_loader)[0]['val_accuracy_epoch']
    log.info(f"Model accuracy: {accuracy}")
    return accuracy


if __name__ == "__main__":
    L.seed_everything(42)

    # parser = ArgumentParser()
    # parser = L.Trainer.add_argparse_args(parser)
    # parser.add_argument("--project_name", default='cc', type=str)
    # parser.add_argument("--data_folder", default='data',type=str)
    # parser.add_argument("--n_layer", default=2,type=int)
    # parser.add_argument("--n_head", default=1,type=int)
    # parser.add_argument("--n_embd", default=128,type=int)
    # parser.add_argument("--n_hidden", default=16,type=int)
    # parser.add_argument("--tied", type=bool)
    # parser.add_argument("--num_epochs", default=1, type=int)
    # parser.add_argument("--learning_rate", default=3e-4, type=float)
    # parser.add_argument("--wdecay", default=1.2e-6, type=float)
    # parser.add_argument("--block_size", default=32, type=int)
    # parser.add_argument("--batch_size", default=64, type=int)
    # parser.add_argument("--num_workers", default=4, type=int)
    #args = parser.parse_args()

    main()

    # uložit a načíst model, když dám jeden example tak chci vidět attention mapu, kterou vytvoří první attention vrstva.
    # přes forward hook