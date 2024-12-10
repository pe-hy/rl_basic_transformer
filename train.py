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
class LitLLM(L.LightningModule):
    def __init__(self, checkpoint_dir, tokenizer_dir=None, trainer_ckpt_path=None):
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


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    batch_size = cfg.model.batch_size
    accumulate_grad_batches = cfg.model.accumulate_grad_batches
    num_workers = cfg.data.num_workers

    lit_model = LitLLM(checkpoint_dir=cfg.model.checkpoint_dir, tokenizer_dir=cfg.model.tokenizer_path)
    lit_model.llm.preprocessor.tokenizer = get_tokenizer(cfg.tok_data)
    tokenizer = lit_model.llm.preprocessor.tokenizer
    datasets = get_data(cfg, tokenizer)

    data = Datamodule(datasets, batch_size, num_workers, tokenizer)

    data.connect(max_seq_length=cfg.model.block_size)

    logger = WandbLogger(project="sos", name=f"{cfg.model.name}")

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

    trainer = L.Trainer(
        devices=2,
        accelerator="cuda",
        max_epochs=20,
        accumulate_grad_batches=accumulate_grad_batches,
        precision="bf16-true",
        val_check_interval=1.0,
        callbacks=[eval_callback],
        logger=logger,
    )
    trainer.fit(lit_model, data)

    lit_model.llm.model.to(lit_model.llm.preprocessor.device)
    lit_model.llm.save(cfg.convert_hf.in_path)


if __name__ == "__main__":
    main()