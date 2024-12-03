import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
import torch.optim
from lightning import Trainer
import hydra
from omegaconf import DictConfig, OmegaConf
from model.pl_module import Pl_model_wrapper
from transformers import PreTrainedTokenizerFast
from data.data import *
@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    
    print(OmegaConf.to_yaml(cfg))
    model_name = cfg.model.model_name

    tokenizer = get_tokenizer(cfg.data)
    datasets = get_data(cfg, tokenizer)

    data = Datamodule(datasets, cfg.train.batchsize, cfg.data.num_workers, tokenizer)

    model = Pl_model_wrapper(
        model_config=cfg.model,
        train_config=cfg.train
    )
    
    logger = WandbLogger(project="sos", name=f"{model_name}")
    
    trainer = Trainer(max_epochs=cfg.train.max_epochs, 
                         logger=logger,
                         accelerator="gpu", devices=4, num_nodes=1, strategy="ddp", accumulate_grad_batches=2,
                         gradient_clip_val=cfg.train.grad_clip,
                         precision="bf16")
    
    trainer.fit(model, data)

    trainer.save_checkpoint(cfg.other.ckpt)

    accuracy = trainer.validate(model,data.val_dataloader())
    return accuracy


if __name__ == "__main__":
    main()