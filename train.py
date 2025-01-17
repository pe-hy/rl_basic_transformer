import hydra
import lightning as L
import torch
import torch.optim
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from model.pl_module import Pl_model_wrapper
from data.data import *


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    model_name = cfg.model.model_name

    datasets = get_data(cfg.data)

    data = Datamodule(datasets, cfg.train.batchsize, cfg.data.num_workers)
    model = Pl_model_wrapper(model_config=cfg.model, train_config=cfg.train)

    # logger = WandbLogger(project="GPT2", name=f"{model_name}")

    trainer = Trainer(
        max_epochs=cfg.train.max_epochs,
        # logger=logger,
        accelerator="cuda",
        devices=1,
        gradient_clip_val=cfg.train.grad_clip,
    )

    trainer.fit(model, data)

    trainer.save_checkpoint(cfg.other.ckpt)

    accuracy = trainer.validate(model, data.val_dataloader())
    return accuracy


if __name__ == "__main__":
    main()
