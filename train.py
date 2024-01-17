import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from omegaconf import OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader, Dataset

from dataset import TSPDataset, collate_fn, make_tgt_mask
from model import make_model, subsequent_mask
from loss import SimpleLossCompute, LabelSmoothing


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))


class TSPModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = make_model(
            src_sz=cfg.node_size, 
            tgt_sz=cfg.node_size, 
            N=cfg.num_layers, 
            d_model=cfg.d_model, 
            d_ff=cfg.d_ff, 
            h=cfg.h, 
            dropout=cfg.dropout
        )
        self.automatic_optimization = False
        criterion = LabelSmoothing(size=cfg.node_size, smoothing=cfg.smoothing)
        self.loss_compute = SimpleLossCompute(self.model.generator, criterion, cfg.node_size)
        self.save_hyperparameters(cfg)  # save config file with pytorch lightening
        self.train_outputs = []
        self.val_outputs = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr, betas=self.cfg.betas, eps=self.cfg.eps)
        lr_scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: rate(step, model_size=self.cfg.d_model, factor=self.cfg.factor, warmup=self.cfg.warmup),
        )
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def train_dataloader(self):
        train_dataset = TSPDataset(self.cfg.train_data_path)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size = self.cfg.train_batch_size, 
            shuffle = True, 
            collate_fn = collate_fn,
            pin_memory=True
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataset = TSPDataset(self.cfg.val_data_path)
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size = self.cfg.val_batch_size, 
            shuffle = False, 
            collate_fn = collate_fn,
            pin_memory=True
        )
        return val_dataloader
    
    def training_step(self, batch):
        src = batch["src"]
        tgt = batch["tgt"]
        visited_mask = batch["visited_mask"]
        tgt_y = batch["tgt_y"]
        ntokens = batch["ntokens"]
        tgt_mask = batch["tgt_mask"]

        opt = self.optimizers() # manual backprop
        opt.zero_grad() # manual backprop
        
        self.model.train()
        out = self.model(src, tgt, visited_mask, tgt_mask) # [B, V, E]
        loss = self.loss_compute(out, tgt, tgt_y, ntokens) # [B, V, E], [B, V]

        training_step_outputs = [l.item() for l in loss]
        self.train_outputs.extend(training_step_outputs)

        loss = loss.mean()
        self.manual_backward(loss) # manual backprop
        opt.step() # manual backprop

        if self.trainer.is_global_zero:
            self.log(
                name = "train_loss",
                value = loss,
                prog_bar = True,
            )
        
        return {"loss": loss}

    def on_train_epoch_start(self) -> None:
        if self.trainer.is_global_zero:
            self.train_start_time = time.time()
    
    def on_train_epoch_end(self):
        outputs = self.all_gather(self.train_outputs)
        self.train_outputs.clear()
        
        lr_scheduler = self.lr_schedulers() # manual backprop
        lr_scheduler.step() # manual backprop
        
        if self.trainer.is_global_zero:
            train_loss = torch.stack(outputs).mean()
            train_time = time.time() - self.train_start_time
            self.print(
                f"Epoch {self.current_epoch}: ",
                "train_loss={:.03f}, ".format(train_loss),
                "train time={:.03f}".format(train_time),
            )
            
    def validate_all(self, batch):
        src = batch["src"]
        tgt = batch["tgt"]
        visited_mask = batch["visited_mask"]
        tgt_y = batch["tgt_y"]
        ntokens = batch["ntokens"]
        tgt_mask = batch["tgt_mask"]
        
        self.model.eval()
        out = self.model(src, tgt, visited_mask, tgt_mask)
        loss = self.loss_compute(out, tgt, tgt_y, ntokens) # [B, V, E], [B, V]
        
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self.validate_all(batch)

        validation_step_outputs = [l.item() for l in loss]
        self.val_outputs.extend(validation_step_outputs)
        return {"loss": loss.mean()}

    def on_validation_epoch_start(self) -> None:
        if self.trainer.is_global_zero:
            self.validation_start_time = time.time()

    def on_validation_epoch_end(self):
        outputs = self.all_gather(self.val_outputs)
        self.val_outputs.clear()
        val_loss = torch.stack(outputs).mean()
        
        self.log(
            name = "val_loss",
            value = val_loss,
            prog_bar = True,
        )
        
        if self.trainer.is_global_zero:
            validation_time = time.time() - self.validation_start_time
            self.print(
                f"\nEpoch {self.current_epoch}: ",
                "val_loss={:.03f}, ".format(val_loss),
                "validation time={:.03f}".format(validation_time),
            )
            
if __name__ == "__main__":
    cfg = OmegaConf.create({
        "train_data_path": "./tsp20_test_concorde.txt",
        "val_data_path": "./tsp20_test_concorde.txt",
        "node_size": 20,
        "train_batch_size": 64,
        "val_batch_size": 64,
        "resume_checkpoint": None,
        "gpus": [0, 1, 2, 3],
        "max_epochs": 20,
        "num_layers": 6,
        "d_model": 128,
        "d_ff": 512,
        "h":8,
        "dropout": 0.1,
        "smoothing": 0.1,
        "seed": 1,
        "lr": 0.5, 
        "betas": (0.9, 0.98),
        "eps": 1e-9,
        "factor": 1.0,
        "warmup": 400,
    })
    pl.seed_everything(cfg.seed)
    tsp_model = TSPModel(cfg)

    checkpoint_callback = ModelCheckpoint(
        monitor = "val_loss",
        filename = f'TSP{cfg.node_size}-' + "{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        mode="min",
        every_n_epochs=1,
    )

    loggers = []
    tb_logger = TensorBoardLogger("logs")
    
    # build trainer
    trainer = pl.Trainer(
        default_root_dir="./",
        devices=cfg.gpus,
        accelerator="cuda",
        precision="16-mixed",
        max_epochs=cfg.max_epochs,
        reload_dataloaders_every_n_epochs=0,
        num_sanity_val_steps=0,
        enable_checkpointing=cfg.resume_checkpoint,
        logger=[tb_logger],
        callbacks=[checkpoint_callback],
        # strategy="ddp_find_unused_parameters_true",
    )

    # training and save ckpt
    trainer.fit(tsp_model)