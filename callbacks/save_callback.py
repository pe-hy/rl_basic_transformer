import lightning.pytorch as L
from lightning.pytorch.utilities import rank_zero_only
class SaveBeforeEvalCallback(L.Callback):
    def __init__(self, save_path: str, eval_interval: int):
        self.save_path = save_path
        self.eval_interval = eval_interval
    
    @rank_zero_only
    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs, batch, batch_idx):
        """ Save the model before eval_callback is triggered """
        current_step = trainer.global_step
        if current_step % self.eval_interval == 0:
            print(f"Saving model at step {current_step} before evaluation...")
            pl_module.llm.model.to(pl_module.llm.preprocessor.device)
            pl_module.llm.save(self.save_path)
