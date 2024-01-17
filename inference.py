import time
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from omegaconf import OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader, Dataset

from dataset_inference import TSPDataset, collate_fn, make_tgt_mask
from model import make_model, subsequent_mask
from loss import SimpleLossCompute, LabelSmoothing

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
        criterion = LabelSmoothing(size=cfg.node_size, smoothing=cfg.smoothing)
        self.loss_compute = SimpleLossCompute(self.model.generator, criterion, cfg.node_size)
        self.save_hyperparameters(cfg)  # save config file with pytorch lightening
        self.val_outputs = []
        self.test_outputs = []

    def test_dataloader(self):
        test_dataset = TSPDataset(self.cfg.val_data_path)
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size = self.cfg.val_batch_size, 
            shuffle = False, 
            collate_fn = collate_fn,
            pin_memory=True
        )
        return test_dataloader

    def test_step(self, batch, batch_idx):
        src = batch["src"]
        tgt = batch["tgt"]
        visited_mask = batch["visited_mask"]
        ntokens = batch["ntokens"]
        tgt_mask = batch["tgt_mask"]
        tsp_tours = batch["tsp_tours"]
        
        batch_size = tsp_tours.shape[0]
        
        self.model.eval()
        with torch.no_grad():
            memory = self.model.encode(src)
            ys = tgt.clone()
            visited_mask = visited_mask.clone()
            for i in range(self.cfg.node_size - 1):
                # memory, tgt, tgt_mask
                tgt_mask = subsequent_mask(ys.size(1)).type(torch.bool).to(src.device)
                out = self.model.decode(memory, src, ys, visited_mask, tgt_mask)
                prob = self.model.generator(out[:, -1], visited_mask)
                _, next_word = torch.max(prob, dim=1)
                
                visited_mask[torch.arange(batch_size), next_word] = True
                ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
        
        total = reduce((lambda x, y: x * y), ys.shape)
        correct = (ys == tsp_tours).sum()
        print(ys)
        print()
        result = {"correct": correct, "total": total}
        
        self.test_outputs.append(result)
        
        return result
        
    def on_test_epoch_end(self):
        outputs = self.all_gather(self.test_outputs)
        self.test_outputs.clear()
        
        if self.trainer.is_global_zero:
            correct = torch.stack([x['correct'] for x in outputs]).sum()
            total = torch.stack([x['total'] for x in outputs]).sum()
            accuracy = (correct / total) * 100
            
            self.print(
                f"correct={correct}",
                f"total={total}",
                f"accuracy={accuracy}"
            )


if __name__ == "__main__":
    cfg = OmegaConf.create({
        "train_data_path": "./tsp20_test_concorde.txt",
        "val_data_path": "./tsp20_test_concorde.txt",
        "node_size": 20,
        "train_batch_size": 16,
        "val_batch_size": 16,
        "resume_checkpoint": "/home/CycleFormer/logs/lightning_logs/version_0/checkpoints/TSP50-epoch=11-val_loss=7629.5786.ckpt",
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
    # tsp_model = TSPModel.load_from_checkpoint(cfg.resume_checkpoint)
    
    # build trainer
    trainer = pl.Trainer(
        default_root_dir="./",
        devices=cfg.gpus,
        accelerator="cuda",
        precision="16-mixed",
        max_epochs=cfg.max_epochs,
        reload_dataloaders_every_n_epochs=0,
        num_sanity_val_steps=0,
    )

    trainer.test(tsp_model)