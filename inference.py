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
        criterion = LabelSmoothing(size=cfg.node_size, smoothing=cfg.smoothing)
        self.loss_compute = SimpleLossCompute(self.model.generator, criterion, cfg.node_size)
        self.save_hyperparameters(cfg)  # save config file with pytorch lightening
        self.val_outputs = []
        self.test_outputs = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr, betas=self.cfg.betas, eps=self.cfg.eps)
        lr_scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: rate(step, model_size=self.cfg.d_model, factor=self.cfg.factor, warmup=self.cfg.warmup),
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"}]

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

        self.model.train()
        out = self.model(src, tgt, visited_mask, tgt_mask) # [B, V, E]
        loss, loss_node = self.loss_compute(out, tgt_y, ntokens) # [B, V, E], [B, V]
        loss = loss.mean().item()
        loss_node = loss_node.mean()

        self.log(
            name="train_loss",
            value=loss,
            prog_bar=True,
        )

        assert torch.isnan(loss_node).sum() == 0, print("loss_node is nan!")

        return {"loss": loss_node}

    def train_epoch_end(self, outputs):
        outputs = torch.as_tensor([output["loss"] for output in outputs])
        self.train_loss_mean = outputs.mean().item()

    def validate_all(self, batch):
        src = batch["src"]
        tgt = batch["tgt"]
        visited_mask = batch["visited_mask"]
        tgt_y = batch["tgt_y"]
        ntokens = batch["ntokens"]
        tgt_mask = batch["tgt_mask"]
        
        self.model.eval()
        out = self.model(src, tgt, visited_mask, tgt_mask)
        loss, loss_node = self.loss_compute(out, tgt_y, ntokens)
        loss = loss.mean().item()
        loss_node = loss_node.mean()

        self.log(
            name="val_loss",
            value=loss,
            prog_bar=True,
        )

        assert torch.isnan(loss_node).sum() == 0, print("loss_node is nan!")

        return {"loss": loss_node}

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            output = self.validate_all(batch)
        self.val_outputs.append(output)
        return output

    def validation_step_end(self, batch_parts):
        return batch_parts

    def on_validation_epoch_start(self) -> None:
        self.validation_start_time = time.time()

    # TODO: all gather로 여러 gpu의 결과 모아서 구현하기
    def on_validation_epoch_end(self):
        loss = [item["loss"] for item in self.val_outputs]
        self.val_outputs = []

        validation_time = time.time() - self.validation_start_time
        val_loss = sum(loss) / len(loss)
        self.log_dict(
            {
                "val_loss(epoch)": val_loss
            },
            on_epoch=True,
            on_step=False,
        )

        self.print(
            f"\nEpoch {self.current_epoch}: ",
            "val_loss(epoch)={:.03f}, ".format(val_loss),
            "validation time={:.03f}".format(validation_time),
        )

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
        "train_data_path": "./tsp50_test_concorde.txt",
        "val_data_path": "./tsp50_test_concorde.txt",
        "node_size": 50,
        "train_batch_size": 4,
        "val_batch_size": 4,
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
    tsp_model = TSPModel.load_from_checkpoint(cfg.resume_checkpoint)
    
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
    )

    trainer.test(tsp_model)