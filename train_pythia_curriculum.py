# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import torch
from litgpt import LLM
from litgpt.data import Alpaca2k
import lightning as L
from utils.data_pythia import *
import hydra
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from utils.curr_evaluator import *
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from config import hf_config
from litgpt.config import configs, Config, name_to_config
from litgpt.model import GPT
from litgpt.api import Preprocessor
import json
import os
import logging


class LitLLM(L.LightningModule):
    def __init__(self, cfg, model, preprocessor, stage):
        super().__init__()
        self.llm = model
        self.cfg = cfg
        self.preprocessor = preprocessor
        self.stage_num = stage
        _, self.hf_conf = hf_config.get_configs(cfg)
        self.total_steps = 0  # Global step count
        self.global_epoch = 0  # Global epoch count
        self.batches_per_epoch = 0

    def on_train_epoch_end(self):
        """Increment global_epoch by 1 after every epoch (even across stages)."""
        self.global_epoch += 1

    def save_training_state(self, stage):
        checkpoint = {
            "optimizer": self.trainer.optimizers[0].state_dict(),
            "scheduler": self.trainer.lr_scheduler_configs[0].scheduler.state_dict(),
            "stage": stage,
            "total_steps": self.total_steps,
            "global_epoch": self.global_epoch,
        }
        save_path = os.path.join(
            self.cfg.convert_hf.in_path, f"stage_{stage}_training_state.pt"
        )
        torch.save(checkpoint, save_path)

    def load_training_state(self, stage):
        load_path = os.path.join(
            self.cfg.convert_hf.in_path, f"stage_{stage-1}_training_state.pt"
        )
        if os.path.exists(load_path):
            checkpoint = torch.load(load_path)
            self.trainer.optimizers[0].load_state_dict(checkpoint["optimizer"])
            self.trainer.lr_scheduler_configs[0].scheduler.load_state_dict(
                checkpoint["scheduler"]
            )
            self.total_steps = checkpoint["total_steps"]
            self.global_epoch = checkpoint["global_epoch"]

    def on_validation_epoch_end(self):
        save_path = os.path.join(self.cfg.convert_hf.in_path, f"stage_{self.stage_num}")
        self.llm.model.to(self.llm.preprocessor.device)
        self.llm.save(save_path)

        self.llm.model.to(self.device)
        current_total_step = (
            self.total_steps  # Steps from previous stages
            + (self.trainer.current_epoch + 1)
            * self.batches_per_epoch  # Include current epoch
        )
        self.log(
            "trainer/eval_folder_step",
            current_total_step,
            on_epoch=True,
            sync_dist=True,
        )
        self.evaluator = CountdownEvaluator(
            config=self.cfg,
            stage=self.stage_num,
            tokenizer=self.preprocessor.tokenizer,
            step=current_total_step,
        )
        metrics = self.evaluator.evaluate()
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.log(f"{k}", v, on_epoch=True, sync_dist=True)

    def setup(self, stage):
        save_path = os.path.join(self.cfg.convert_hf.in_path, f"stage_{self.stage_num}")
        self.preprocessor.tokenizer.save_pretrained(save_path)
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(self.hf_conf, f, indent=2)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        idx, targets, att_mask = (
            batch["input_ids"],
            batch["labels"],
            batch["attention_mask"],
        )
        _, loss = self(idx, targets)
        self.log("train_loss", loss, sync_dist=True)

        current_total_step = (
            self.total_steps  # Steps from previous stages
            + self.trainer.current_epoch
            * self.batches_per_epoch  # Epochs completed in this stage
            + batch_idx  # Current batch in this epoch
        )

        self.log("trainer/total_step", current_total_step, sync_dist=True)
        self.log("total_epoch", self.global_epoch, sync_dist=True)

        if self.total_steps % 10 == 0:  # Every 100 steps
            print(f"\nCurrent LR: {self.trainer.optimizers[0].param_groups[0]['lr']}\n")

        return loss

    def validation_step(self, batch, batch_idx):
        # if self.global_step == 0:
        #     # print(batch["input_ids"])
        #     # print(batch["input_ids"].shape)
        idx, targets, att_mask = (
            batch["input_ids"],
            batch["labels"],
            batch["attention_mask"],
        )
        logits, loss = self(idx, targets)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        return {"val_loss": loss}

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.llm(idx, targets)

    def configure_optimizers(self):
        warmup_steps = 10
        base_lr = 0.0002

        # Start with very small learning rate
        optimizer = torch.optim.AdamW(
            [{"params": self.llm.model.parameters(), "initial_lr": base_lr}],
            lr=base_lr,
            weight_decay=0.0,
            betas=(0.9, 0.95),
        )

        def lr_lambda(step):
            stage_step = step + (self.trainer.current_epoch * self.batches_per_epoch)
            # If we want to reach multiplier of 9.9 over total_steps
            effective_warmup = (55 * self.batches_per_epoch) / 9.9
            return stage_step / effective_warmup

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda,
            last_epoch=-1,  # Always start fresh for scheduler counting
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update LR every step
            },
        }


@hydra.main(
    config_path="config",
    config_name="config_pythia_curriculum",
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
    curriculum_datasets = get_curriculum(cfg, tokenizer)

    num_stages = len(curriculum_datasets)
    epochs_per_stage = [min(stage + 1, 10) for stage in range(num_stages)]
    print(
        "Number of datasets in cur: ",
        num_stages,
        "total epochs: ",
        sum(epochs_per_stage),
    )

    model = LLM(GPT(conf), preprocessor=preprocessor, config=conf)
    lit_model = LitLLM(model=model, cfg=cfg, preprocessor=preprocessor, stage=0)

    logger = WandbLogger(
        project="sos",
        name=f"{cfg.model.name}",
        id="564165x",
        resume="allow",
        config=wandb_config,
    )
    for stage, data in enumerate(curriculum_datasets):
        lit_model.stage_num = stage
        print("lit_model.stage: ", lit_model.stage_num, "stage: ", stage)
        # print(tokenizer.decode(data["train"][sample_idx]["input_ids"]))
        wandb_config.update({"curriculum_stage": stage})
        logger.experiment.config.update(
            {"curriculum_stage": stage}, allow_val_change=True
        )

        if stage > 0:
            lit_model.load_training_state(stage)

        current_epochs = epochs_per_stage[stage]
        data = Datamodule(
            dataset=data,
            batch_size=batch_size,
            num_workers=num_workers,
            tokenizer=tokenizer,
        ).connect(max_seq_length=cfg.model.block_size)

        data.setup()

        batches_per_epoch = len(data.train_dataloader())
        print("batches per epoch:", batches_per_epoch)
        total_batches_this_stage = batches_per_epoch * current_epochs
        lit_model.batches_per_epoch = batches_per_epoch

        if stage == len(curriculum_datasets) - 1:
            checkpoint_callback = ModelCheckpoint(
                monitor="countdown_eval/accuracy",
                dirpath=f"temp/{cfg.model.name}/checkpoints/stage_{stage}",
                filename="{epoch:02d}-{countdown_eval-accuracy:.4f}",
                save_top_k=2,
                mode="max",
            )
        else:
            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss",
                dirpath=f"temp/{cfg.model.name}/checkpoints/stage_{stage}",
                filename="{epoch:02d}-{val_loss:.4f}",
                save_top_k=2,
                mode="min",
            )
        trainer = L.Trainer(
            devices=1,
            accelerator="cuda",
            max_epochs=current_epochs,
            accumulate_grad_batches=accumulate_grad_batches,
            precision="bf16-true",
            val_check_interval=1.0,
            callbacks=[
                checkpoint_callback,
                LearningRateMonitor(),
            ],
            logger=logger,
        )
        trainer.fit(lit_model, data)
        lit_model.total_steps += batches_per_epoch * current_epochs
        lit_model.save_training_state(stage)

    final_save_path = os.path.join(cfg.convert_hf.in_path, f"stage_{num_stages-1}")
    lit_model.llm.model.to(lit_model.llm.preprocessor.device)
    lit_model.llm.save(final_save_path)


if __name__ == "__main__":
    main()
