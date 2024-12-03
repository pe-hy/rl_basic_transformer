import lightning as L
import torch
from typing import Optional
import torch.optim
from lightning import LightningModule
from models.models import GPT
from omegaconf import DictConfig

class Pl_model_wrapper(LightningModule):
    def __init__(
        self,
        model_config: DictConfig,
        train_config: DictConfig
    ):
        super().__init__()
        self.betas = train_config.betas
        self.weight_decay = train_config.weight_decay
        self.learning_rate = train_config.learning_rate
        self.patience = train_config.patience
        self.model_config = model_config
        self.save_hyperparameters()
        self.gpt = GPT(model_config)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.gpt(idx, targets)

    def configure_optimizers(self) -> dict:
        optimizer = self.gpt.configure_optimizers(
            weight_decay=self.weight_decay,
            learning_rate=self.learning_rate,
            betas=self.betas
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1.e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss"
            }
        }
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        idx, targets = batch
        _, loss = self(idx, targets)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        idx, targets = batch
        logits, loss = self(idx, targets)
        #accuracy = self.calculate_accuracy(logits, targets)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        #self.log('val_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)    
        return {'val_loss': loss}

    def calculate_accuracy(self, logits, targets):
        pass
        #predictions = torch.argmax(logits, dim=-1)
        #targets = targets[:, 0] # prev_enabling, enabling, subgoal, prev_start, prev_subgoal
        #return (predictions == targets).float().mean()