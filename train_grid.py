import argparse

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

from dataset_grid_train import TSPDataset, collate_fn, make_tgt_mask
from dataset_grid_inference import TSPDataset as TSPDataset_Val
from dataset_grid_inference import collate_fn as collate_fn_val

from model import make_model, subsequent_mask
from loss import SimpleLossComputeWithMask, LabelSmoothingWithMask

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
        self.model = make_model(
            grid_size = cfg.grid_size,
            N=cfg.num_layers, 
            d_model=cfg.d_model, 
            d_ff=cfg.d_ff, 
            h=cfg.h, 
            dropout=cfg.dropout,
            encoder_pe_option = cfg.encoder_pe_option,
            decoder_pe_option = cfg.decoder_pe_option,
            use_start_token = cfg.use_start_token,
            share_lut = cfg.share_lut,
            node_size = cfg.node_size + 1 if cfg.use_start_token else cfg.node_size
        )
        self.automatic_optimization = False
        
        if cfg.use_start_token:
            criterion = LabelSmoothingWithMask(size = cfg.grid_size * cfg.grid_size + 1 , smoothing=cfg.smoothing)
        else:
            criterion = LabelSmoothingWithMask(size = cfg.grid_size * cfg.grid_size , smoothing=cfg.smoothing)
        self.loss_compute = SimpleLossComputeWithMask(self.model.generator, criterion)
    
        self.set_cfg(cfg)
        self.train_outputs = []
        
        self.val_optimal_tour_distances = []
        self.val_predicted_tour_distances = []
        
    def set_cfg(self, cfg):
        self.cfg = cfg
        self.save_hyperparameters(cfg)  # save config file with pytorch lightening

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr, betas=self.cfg.betas, eps=self.cfg.eps)
        lr_scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: rate(step, model_size=self.cfg.d_model, factor=self.cfg.factor, warmup=self.cfg.warmup),
        )
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def train_dataloader(self):
        train_dataset = TSPDataset(self.cfg.train_data_path, self.cfg.grid_size, cfg.use_start_token)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size = self.cfg.train_batch_size, 
            shuffle = True, 
            collate_fn = collate_fn,
            pin_memory=True
        )
        return train_dataloader

    def val_dataloader(self):
        self.val_dataset = TSPDataset_Val(self.cfg.val_data_path, self.cfg.grid_size, cfg.use_start_token)
        val_dataloader = DataLoader(
            self.val_dataset, 
            batch_size = self.cfg.val_batch_size, 
            shuffle = False, 
            collate_fn = collate_fn_val,
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
        out = self.model(src, tgt, tgt_mask) # [B, V, E]
        
        loss = self.loss_compute(out, tgt_y, visited_mask > 0, ntokens) # [B, V, E], [B, V]

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
                f"##############Train: Epoch {self.current_epoch}###################\n",
                "train_loss={:.03f}, ".format(train_loss),
                "train time={:.03f}\n".format(train_time),
                f"##################################################################\n"
            )

    def validation_step(self, batch, batch_idx):
        src = batch["src"]
        tgt = batch["tgt"]
        visited_mask = batch["visited_mask"]
        ntokens = batch["ntokens"]
        tgt_mask = batch["tgt_mask"]
        tsp_tours = batch["tsp_tours"]
        
        batch_size = tsp_tours.shape[0]
        decode_step = self.cfg.node_size if cfg.use_start_token else self.cfg.node_size - 1
        
        self.model.eval()
        with torch.no_grad():
            memory = self.model.encode(src)
            ys = tgt.clone()
            visited_mask = visited_mask.clone()
            for i in range(decode_step):
                # memory, tgt, tgt_mask
                tgt_mask = subsequent_mask(ys.size(1)).type(torch.bool).to(ys.device)
                out = self.model.decode(memory, ys, tgt_mask)
                prob = self.model.generator(out[:, -1], visited_mask > 0)
                _, next_word = torch.max(prob, dim=1)
                
                visited_mask[torch.arange(batch_size), next_word] = visited_mask[torch.arange(batch_size), next_word] + 1
                ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
        
            if cfg.use_start_token:
                ys = ys[:, 1:]
        
        optimal_tour_distance = self.get_tour_distance(tsp_tours)
        predicted_tour_distance = self.get_tour_distance(ys)
        
        result = {
            "optimal_tour_distance": optimal_tour_distance.tolist(),
            "predicted_tour_distance": predicted_tour_distance.tolist(),
            }    
        
        self.val_optimal_tour_distances.extend(result["optimal_tour_distance"])
        self.val_predicted_tour_distances.extend(result["predicted_tour_distance"])
            
        return result
    
    def get_tour_distance(self, ordered_seq):
        x_coordinates = (ordered_seq % self.cfg.grid_size) / self.cfg.grid_size  + 1 / (2 * self.cfg.grid_size)
        y_coordinates = (ordered_seq // self.cfg.grid_size) / self.cfg.grid_size  + 1 / (2 * self.cfg.grid_size)
        coordinates = torch.stack([x_coordinates, y_coordinates], dim=-1)
        rolled_coordinates = coordinates.roll(dims = 1, shifts = -1)
        segment_lengths = ((coordinates - rolled_coordinates) ** 2).sum(-1).sqrt() # [B, N]
        group_travel_distances = segment_lengths.sum(-1)
        return group_travel_distances

    def on_validation_epoch_start(self) -> None:
        if self.trainer.is_global_zero:
            self.validation_start_time = time.time()

    def on_validation_epoch_end(self):
        optimal_tour_distances = self.all_gather(sum(self.val_optimal_tour_distances))
        predicted_tour_distances = self.all_gather(sum(self.val_predicted_tour_distances))
        
        self.val_optimal_tour_distances.clear()
        self.val_predicted_tour_distances.clear()
        
        mean_optimal_tour_distance = optimal_tour_distances.sum().item() / len(self.val_dataset)
        mean_predicted_tour_distance = predicted_tour_distances.sum().item() / len(self.val_dataset)
        mean_opt_gap = (mean_predicted_tour_distance - mean_optimal_tour_distance) / mean_optimal_tour_distance * 100
        
        self.log(
            name = "opt_gap",
            value = mean_opt_gap,
            prog_bar = True,
            sync_dist=True
        )
        
        if self.trainer.is_global_zero:
            validation_time = time.time() - self.validation_start_time
            self.print(
                f"##############Validation: Epoch {self.current_epoch}##############",
                "validation time={:.03f}".format(validation_time),
                f"\nmean_optimal_tour_distance = {mean_optimal_tour_distance}",
                f"\nmean_predicted_tour_distance = {mean_predicted_tour_distance}",
                f"\nmean_opt_gap = {mean_opt_gap}  %",
                f"##################################################################\n",
            )
            
def check_cfg(cfg):
    if not cfg.encoder_pe_option in ["pe_2d", None]:
        return False
    
    if not cfg.decoder_pe_option in ["pe_2d", "pe_1d_original", "pe_1d_learnable", "pe_1d_circular", "pe_1d_learnable_circular", None]:
        return False
    
    if cfg.use_start_token:
        if cfg.decoder_pe_option == "pe_2d":
            return False
        
    return True

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config.yaml", help="Path to the configuration YAML file.")
    args = parser.parse_args()
    return args
        
if __name__ == "__main__":
    args = parse_arguments()
    cfg = OmegaConf.load(args.config)
    
    if not check_cfg(cfg):
        raise ValueError("Check config again!")
    
    pl.seed_everything(cfg.seed)
    tsp_model = TSPModel(cfg)

    checkpoint_callback = ModelCheckpoint(
        monitor = "opt_gap",
        filename = f'TSP{cfg.node_size}-' + "{epoch:02d}-{opt_gap:.4f}",
        save_top_k=1,
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
        num_sanity_val_steps=2,
        enable_checkpointing=cfg.resume_checkpoint,
        logger=[tb_logger],
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch = 1
        # strategy="ddp_find_unused_parameters_true",
    )

    # training and save ckpt
    trainer.fit(tsp_model)