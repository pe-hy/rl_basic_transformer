# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import torch
from litgpt import LLM
from litgpt.data import Alpaca2k
import lightning as L
from utils.data_pythia import *
import hydra
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from callbacks.eval_callback import EvalCallback
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from config import hf_config
from litgpt.config import configs, Config, name_to_config
from litgpt.model import GPT
from litgpt.api import Preprocessor

import json
import os


class LitLLM(L.LightningModule):
    def __init__(self, cfg, model, preprocessor, trainer_ckpt_path=None):
        super().__init__()

        # self.llm = LLM.load(
        #     checkpoint_dir, tokenizer_dir=tokenizer_dir, distribute=None
        # )
        # return cls(
        #     model=model, preprocessor=preprocessor, prompt_style=prompt_style,
        #     config=config, checkpoint_dir=checkpoint_dir, fabric=fabric, generate_strategy=None,
        #     kv_cache_initialized=False, fixed_kv_cache_size=False
        # )

        self.llm = model
        self.cfg = cfg
        self.preprocessor = preprocessor
        self.trainer_ckpt_path = trainer_ckpt_path
        _, self.hf_conf = hf_config.get_configs(cfg)

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

    def validation_step(self, batch, batch_idx):
        idx, targets, att_mask = (
            batch["input_ids"],
            batch["labels"],
            batch["attention_mask"],
        )
        logits, loss = self(idx, targets)
        # accuracy = self.calculate_accuracy(logits, targets)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        # self.log('val_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return {"val_loss": loss}

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


@hydra.main(config_path="config", config_name="config_karolina_single_run_pythia", version_base=None)
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

    lit_model = LitLLM(model=model, cfg=cfg, preprocessor=preprocessor)
    datasets = get_data(cfg, tokenizer)
    data = Datamodule(datasets, batch_size, num_workers, tokenizer)

    for data in datasets:
        logging.info("#"*10)
        logging.info(f"Curriculum {cfg.cur.type}")
        logging.info(f"Data size: {len(data['train'])}")
        # data = Datamodule(datasets, batch_size, num_workers, tokenizer)
        data = Datamodule(
            data=data,
            batch_size=batch_size,
            num_workers=cfg.data.num_workers,
            tokenizer=tokenizer
        )

    data.connect(max_seq_length=cfg.model.block_size)

    logger = WandbLogger(project="sos", name=f"{cfg.model.name}", config=wandb_config)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # what metric to track
        dirpath=f"temp/{cfg.model.name}/checkpoints",  # where to save checkpoints
        filename="{epoch:02d}-{val_loss:.4f}",  # how to name checkpoints
        save_top_k=2,  # save top 3 models
        mode="min",  # lower val_loss is better
    )

    eval_callback = EvalCallback(
        data_dir=cfg.data.datapath,
        eval_data=cfg.data.val_file,
        tokenizer=tokenizer,
        num_examples=cfg.eval.num_examples,
        batch_size=cfg.eval.batch_size,
        config=cfg,
        eval_interval=cfg.eval.eval_interval,
        save_path=cfg.convert_hf.in_path,
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
        callbacks=[LearningRateMonitor(), EarlyStopping(monitor="countdown_eval/accuracy",
                        patience=cfg.cur.max_epochs,
                        check_on_train_epoch_end = False,
                        stopping_threshold = 0.7,
                        mode = 'max'), checkpoint_callback, eval_callback],
        logger=logger,
    )
    trainer.fit(lit_model, data)

    lit_model.llm.model.to(lit_model.llm.preprocessor.device)
    lit_model.llm.save(cfg.convert_hf.in_path)


if __name__ == "__main__":
    main()
