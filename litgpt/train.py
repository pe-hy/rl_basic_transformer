# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import torch
from litgpt import LLM
from litgpt.data import Alpaca2k
import lightning as L
from data.data import *

from lightning.pytorch.loggers import WandbLogger

from omegaconf import DictConfig, OmegaConf


cfg = OmegaConf.create({
    "data": {
        "datapath": "data",
        "specific_datapath": "sos/",
        "train_file": "sos/train1_b4_t30_n200000_dfs.json",
        "val_target_file": "sos/val_target1_b4_t30_n200000_dfs.json", 
        "val_file": "sos/val1_b4_t30_n200000_dfs.json",
        "tokenizer_path": "data/sos/tokenizer/tokenizer.json",
        "num_train": 2e5,
        "num_workers": 4
    },
    "model": {
        "model_name": "GPT2",
        "n_layers": 12,
        "n_heads": 12,
        "n_embed": 512,
        "dropout": 0.1,
        "n_hidden": 384,
        "block_size": 4096,
        "vocab_size": 410,
        "bias": True
    },
    "train": {
        "batchsize": 12,
        "max_epochs": 10,
        "grad_clip": 1.0,
        "weight_decay": 0.1,
        "learning_rate": 3e-4,
        "patience": 4,
        "betas": [0.9, 0.95]
    },
    "other": {
        "ckpt": "ckpt.pt"
    }
})

class LitLLM(L.LightningModule):
    def __init__(self, checkpoint_dir, tokenizer_dir="data/sos/tokenizer", trainer_ckpt_path=None):
        super().__init__()

        self.llm = LLM.load(checkpoint_dir, tokenizer_dir=tokenizer_dir, distribute=None)
        self.trainer_ckpt_path = trainer_ckpt_path

    def setup(self, stage):
        self.llm.trainer_setup(trainer_ckpt=self.trainer_ckpt_path)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        idx, targets, att_mask = batch["input_ids"], batch["labels"], batch["attention_mask"]
        _, loss = self(idx, targets)
        self.log("train_loss", loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        idx, targets, att_mask = batch["input_ids"], batch["labels"], batch["attention_mask"]
        logits, loss = self(idx, targets)
        #accuracy = self.calculate_accuracy(logits, targets)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log('val_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)    
        return {'val_loss': loss}

    def configure_optimizers(self):
        warmup_steps = 10
        optimizer = torch.optim.AdamW(self.llm.model.parameters(), lr=0.0002, weight_decay=0.0, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / warmup_steps)
        return [optimizer], [scheduler]
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.llm(idx, targets)


if __name__ == "__main__":

    batch_size = 12
    accumulate_grad_batches = 1

    #########################################################
    # Use case 1: Pretraining from random weights
    #########################################################

    # llm = LLM.load("EleutherAI/pythia-160m", tokenizer_dir="EleutherAI/pythia-160m", init="random")
    # llm.save("pythia-160m-random-weights")
    # del llm

    lit_model = LitLLM(checkpoint_dir="pythia-160m-random-weights", tokenizer_dir="/mnt/raid/data/Hyner_Petr/rl/sos_branch/rl_basic_transformer/litgpt/checkpoints/EleutherAI/pythia-160m")
    tokenizer = get_tokenizer(cfg.data)
    datasets = get_data(cfg, tokenizer)

    data = Datamodule(datasets, cfg.train.batchsize, cfg.data.num_workers, tokenizer)

    data.connect(max_seq_length=4096)
    logger = WandbLogger(project="sos", name="Pythia-160m-rerun")
    trainer = L.Trainer(
        logger=logger,
        devices=2,
        accelerator="cuda",
        max_epochs=10,
        accumulate_grad_batches=accumulate_grad_batches,
        precision="bf16-true",
    )
    trainer.fit(lit_model, data)

    lit_model.llm.model.to(lit_model.llm.preprocessor.device)
    lit_model.llm.save("trained_model")
    del lit_model