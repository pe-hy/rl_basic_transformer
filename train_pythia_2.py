# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import torch
from litgpt import LLM
from litgpt.data import Alpaca2k
import lightning as L
from utils.data_pythia_2 import *
import hydra
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from utils.evaluator_2 import *
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from config import hf_config
from litgpt.config import configs, Config, name_to_config
from litgpt.model import GPT
from litgpt.api import Preprocessor
import math
import json
import os

from transformers import get_cosine_schedule_with_warmup

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Starting training...")


class LitLLM(L.LightningModule):
    def __init__(self, cfg, model, preprocessor, train_batches, trainer_ckpt_path=None):
        super().__init__()

        self.llm = model
        self.cfg = cfg
        self.preprocessor = preprocessor
        self.trainer_ckpt_path = trainer_ckpt_path
        self.train_batches = train_batches
        _, self.hf_conf = hf_config.get_configs(cfg)
        print(train_batches)

    def setup(self, stage):
        self.preprocessor.tokenizer.save_pretrained(self.cfg.convert_hf.in_path)
        with open(os.path.join(self.cfg.convert_hf.in_path, "config.json"), "w") as f:
            json.dump(self.hf_conf, f, indent=2)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        idx, targets, att_mask = (
            batch["input_ids"],
            batch["labels"],
            batch["attention_mask"],
        )
        _, loss = self(idx, targets)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        idx, targets, att_mask = (
            batch["input_ids"],
            batch["labels"],
            batch["attention_mask"],
        )
        logits, loss = self(idx, targets)

        # Log different validation losses based on dataloader index
        if dataloader_idx == 0:
            # Main validation set
            self.log(
                "val_loss/val_loss",  # Changed this
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        elif dataloader_idx == 1:
            # test_first (bin 0) validation
            self.log(
                "val_loss/val_loss_test_first",  # Changed this
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        elif dataloader_idx == 2:
            # test_last (last bin) validation
            self.log(
                "val_loss/val_loss_test_last",  # Changed this
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        return {"val_loss": loss}

    def on_validation_epoch_end(self):

        save_path = self.cfg.convert_hf.in_path
        self.llm.model.to(self.llm.preprocessor.device)
        self.llm.save(save_path)

        self.llm.model.to(self.device)

        # Run evaluation periodically
        if (
            self.current_epoch % self.cfg.eval.eval_interval == 0
            and self.trainer.is_global_zero
        ):
            evaluator = CountdownEvaluator(
                self.cfg, self.preprocessor.tokenizer, self.global_step, self.llm.model
            )

            # Get datasets from the datamodule
            datasets = self.trainer.datamodule.dataset
            metrics = evaluator.evaluate(
                test_all_data=datasets["test_all"],
                test_first_data=datasets["test_first"],
                test_last_data=datasets["test_last"],
            )

        for key, value in metrics.items():
            self.log(
                key, value, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True
            )

    def configure_optimizers(self):
        warmup_steps = 10
        optimizer = torch.optim.AdamW(
            self.llm.model.parameters(), lr=0.0002, weight_decay=0.0, betas=(0.9, 0.95)
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: step / warmup_steps
        )
        return [optimizer], [scheduler]

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.llm(idx, targets)


@hydra.main(
    config_path="config",
    config_name="config_pythia",
    version_base=None,
)
def main(cfg: DictConfig):
    conf, _ = hf_config.get_configs(cfg)

    wandb_config = OmegaConf.to_container(cfg, resolve=True)

    print("Current model configuration:")
    print(f"n_layer: {cfg.model.n_layer}")
    print(f"n_head: {cfg.model.n_head}")
    print(f"n_embd: {cfg.model.n_embd}")
    print(f"Model name: {cfg.model.name}")

    batch_size = cfg.model.batch_size
    accumulate_grad_batches = cfg.model.accumulate_grad_batches
    num_workers = cfg.data.num_workers
    tokenizer = get_tokenizer(cfg.tok_data)
    preprocessor = Preprocessor(
        tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    model = LLM(GPT(conf), preprocessor=preprocessor, config=conf)
    datasets = get_data(cfg, tokenizer, 6)
    data = Datamodule(datasets, batch_size, num_workers, tokenizer)
    data.connect(max_seq_length=cfg.model.block_size)
    data.setup()
    train_size = len(data.train_dataloader())
    lit_model = LitLLM(
        model=model, cfg=cfg, train_batches=train_size, preprocessor=preprocessor
    )

    logger = WandbLogger(
        project="sos_new", name=f"{cfg.model.name}", config=wandb_config
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="countdown_eval/accuracy",  # what metric to track
        dirpath=f"temp/{cfg.model.name}/checkpoints",  # where to save checkpoints
        filename="{epoch:02d}-{countdown_eval-accuracy:.4f}",  # how to name checkpoints
        save_top_k=2,  # save top 3 models
        mode="max",  # lower val_loss is better
    )

    total_params = sum(p.numel() for p in model.parameters())
    print("total number of params:", total_params)

    trainer = L.Trainer(
        devices=1,
        accelerator="cuda",
        max_epochs=cfg.model.epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        precision="bf16-true",
        val_check_interval=1.0,
        callbacks=[LearningRateMonitor(), checkpoint_callback],
        logger=logger,
    )
    trainer.fit(lit_model, data)

    lit_model.llm.model.to(lit_model.llm.preprocessor.device)
    lit_model.llm.save(cfg.convert_hf.in_path)


if __name__ == "__main__":
    main()
