# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import torch
from litgpt import LLM
from litgpt.data import Alpaca2k
import lightning as L
from utils.data import *
import hydra
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from callbacks.eval_callback import EvalCallback
from callbacks.save_callback import SaveBeforeEvalCallback
from config import hf_config
from litgpt.config import configs, Config, name_to_config
from litgpt.model import GPT
from litgpt.api import Preprocessor

import json
import os

from hydra.utils import instantiate, get_class
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from pytorch_lightning import Trainer, seed_everything
from model.logger import setup_wandb, get_git_hash, find_existing_checkpoint
import logging
import os
import getpass
import tempfile

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from transformers import XLNetConfig, BitsAndBytesConfig
from model.past import PASTConfig, PAST
import torch
import math


def nearest_multiple(x):
    if x < 128:
        return 128
    base = 128
    res = base * math.ceil(x / base)
    print(f"updating vocab size {x} to nearest multiple of {base} with {res}")
    return res


class PLModel(LightningModule):
    def __init__(
        self, tokenizer, config, config_optim, eval_fn, **model_kwargs
    ) -> None:
        super().__init__()
        self.config = config
        self.config_optim = config_optim
        self.train_mode = config.mode
        self.eval_fn = eval_fn
        self.eval_mode = config.train_mode
        self.tokenizer = tokenizer
        # Create PASTConfig from the hydra config
        past_config = PASTConfig(
            vocab_size=config.vocab_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            dropout=config.dropout,
            bias=config.bias,
            train_mode=config.train_mode,
            tie_lmhead=config.tie_lmhead,
            attn_sink=config.attn_sink,
            stack_enc_dec=config.stack_enc_dec,
        )

        # Initialize PAST model
        self.model = PAST(past_config)
        self.save_hyperparameters(logger=False)
        # important to load model later
        # avoid saving to logger becase we do that elsewhere

    def forward(self, batch, **model_kwargs):
        # print("Forward batch:", batch)  # Debug print
        out = self.model(
            **batch,
            mode=self.train_mode,
            **model_kwargs,
        )
        return out

    def training_step(self, batch, batch_idx):
        bsz = batch["input_ids"].size(0)
        out = self(batch)
        self.log("train/loss", out.loss, prog_bar=True, batch_size=bsz, sync_dist=True)
        return out.loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        bsz = batch["input_ids"].size(0)
        out = self(batch)

        # Log loss
        metric_name = f"val/loss/{dataloader_idx}" if dataloader_idx > 0 else "val/loss"
        self.log(
            metric_name,
            out.loss,
            prog_bar=True,
            batch_size=bsz,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        if self.eval_fn is not None:
            # Get evaluation metrics
            eval_dict = self.eval_fn(
                self.model,
                batch,
                self.trainer.global_step,
                dataloader_idx=dataloader_idx,
                return_samples=batch_idx == 0,
                mode=self.eval_mode,
            )

            # Log numeric metrics only
            for k, v in eval_dict.items():
                if not k.startswith("SAVE_") and not isinstance(v, (dict, list, str)):
                    k = f"{k}"
                    prog = "acc" in k.lower()
                    self.log(
                        k,
                        v,
                        prog_bar=prog,
                        batch_size=bsz,
                        add_dataloader_idx=False,
                        sync_dist=True,
                    )

            # Handle text samples separately
            if hasattr(self.logger, "log_text") and batch_idx == 0:
                text_data = {
                    k: v for k, v in eval_dict.items() if k.startswith("SAVE_")
                }
                if text_data:
                    self.logger.log_text(
                        f"samples/{dataloader_idx}",
                        columns=list(text_data.keys()),
                        data=list(zip(*text_data.values())),
                    )

        return out.loss

    def configure_optimizers(self):
        optim_name = self.config_optim.optimizer
        weight_decay = self.config_optim.weight_decay
        learning_rate = self.config_optim.learning_rate
        betas = self.config_optim.betas
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        if optim_name == "adamw":
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        elif optim_name == "sgd":
            optimizer = torch.optim.SGD(optim_groups, lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer {optim_name}")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.config_optim.warmup_pct,
            anneal_strategy="cos",
            final_div_factor=25,
        )
        # total steps can be set with trainer.max_steps directly
        # but may break when using max_batches instead
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


@hydra.main(config_path="config", config_name="config_past", version_base=None)
def main(cfg: DictConfig):
    conf, _ = hf_config.get_configs(cfg)
    wandb_config = OmegaConf.to_container(cfg, resolve=True)

    print("Current model configuration:")
    print(f"n_layer: {cfg.model.n_layer}")
    print(f"n_head: {cfg.model.n_head}")
    print(f"n_embd: {cfg.model.n_embd}")
    print(f"Model name: {cfg.model.name}")

    batch_size = cfg.model.batch_size
    val_bsz = cfg.eval.batch_size
    accumulate_grad_batches = cfg.model.accumulate_grad_batches
    num_workers = cfg.data.num_workers
    tokenizer = get_tokenizer(cfg.tok_data)
    # preprocessor = Preprocessor(
    #     tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"
    # )

    # model = instantiate(
    #     cfg.model,
    #     config_optim=cfg.optim,
    #     tokenizer=tokenizer,
    # )

    datasets = get_data(cfg, tokenizer)
    val_bsz = cfg.eval.batch_size
    datamodule = Datamodule(datasets, batch_size, val_bsz, num_workers, tokenizer, cfg)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    model = PLModel(
        tokenizer,
        config=cfg.model,
        config_optim=cfg.optim,
        eval_fn=datamodule.eval_fn,
    )

    # data.connect(max_seq_length=cfg.model.block_size)

    logger = WandbLogger(
        project="sos", name=f"{cfg.model.name}_past", config=wandb_config
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="countdown_eval/accuracy",  # what metric to track
        dirpath=f"temp/{cfg.model.name}/checkpoints",  # where to save checkpoints
        filename="{epoch:02d}-{val_loss:.3f}",  # how to name checkpoints
        save_top_k=2,  # save top 3 models
        mode="max",  # lower val_loss is better
    )

    trainer = Trainer(
        devices=1,
        accelerator="cuda",
        max_epochs=cfg.model.epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        precision="bf16-true",
        val_check_interval=1.0,
        callbacks=[LearningRateMonitor(), checkpoint_callback],
        logger=logger,
        default_root_dir=f"{'temp/' + cfg.model.name}",
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
