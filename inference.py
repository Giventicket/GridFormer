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
        self.set_cfg(cfg)
        self.val_outputs = []
        
        self.test_corrects = []
        self.test_optimal_tour_distances = []
        self.test_predicted_tour_distances = []
        
    def set_cfg(self, cfg):
        self.cfg = cfg
        self.save_hyperparameters(cfg)  # save config file with pytorch lightening

    def test_dataloader(self):
        self.test_dataset = TSPDataset(self.cfg.val_data_path)
        test_dataloader = DataLoader(
            self.test_dataset, 
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
                out = self.model.decode(memory, src, ys, tgt_mask)
                prob = self.model.generator(out[:, -1], visited_mask)
                _, next_word = torch.max(prob, dim=1)
                
                visited_mask[torch.arange(batch_size), next_word] = True
                ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
        
        
        correct = (ys == tsp_tours).sum(-1)
        optimal_tour_distance = self.get_tour_distance(src, tsp_tours)
        predicted_tour_distance = self.get_tour_distance(src, ys)
        
        result = {
            "correct": correct.tolist(),
            "optimal_tour_distance": optimal_tour_distance.tolist(),
            "predicted_tour_distance": predicted_tour_distance.tolist(),
            }    
        
        self.test_corrects.extend(result["correct"])
        self.test_optimal_tour_distances.extend(result["optimal_tour_distance"])
        self.test_predicted_tour_distances.extend(result["predicted_tour_distance"])
        
        if self.trainer.is_global_zero:
            print("predicted tour: ", ys[0].tolist())
            print("optimal tour: ", tsp_tours[0].tolist())
            print("opt, pred tour distance: ", optimal_tour_distance[0].item(), predicted_tour_distance[0].item())
            print("optimality gap: ", ((predicted_tour_distance[0].item() - optimal_tour_distance[0].item()) / optimal_tour_distance[0].item()) * 100, "%")
            print("node prediction [hit ratio]: ", (correct[0].item() / self.cfg.node_size) * 100 , "%")
            print()
        
        return result
    
    def get_tour_distance(self, graph, tour):
        # graph.shape = [B, N, 2]
        # tour.shape = [B, N]

        shp = graph.shape
        gathering_index = tour.unsqueeze(-1).expand(*shp)
        ordered_seq = graph.gather(dim = 1, index = gathering_index)
        rolled_seq = ordered_seq.roll(dims = 1, shifts = -1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(-1).sqrt() # [B, N]
        group_travel_distances = segment_lengths.sum(-1)
        return group_travel_distances

    def flatten_list(self, nested_list):
        flat_list = []
        for sublist in nested_list:
            flat_list.extend(sublist)
        return flat_list
        
    def on_test_epoch_end(self):
        corrects = self.all_gather(self.test_corrects)
        optimal_tour_distances = self.all_gather(self.test_optimal_tour_distances)
        predicted_tour_distances = self.all_gather(self.test_predicted_tour_distances)
        
        self.test_corrects.clear()
        self.test_optimal_tour_distances.clear()
        self.test_predicted_tour_distances.clear()
        
        if self.trainer.is_global_zero:
            corrects = torch.stack(corrects)
            optimal_tour_distances = torch.stack(optimal_tour_distances)
            predicted_tour_distances = torch.stack(predicted_tour_distances)
            
            correct = corrects.sum().item()
            total = self.cfg.node_size * len(self.test_dataset)
            hit_ratio = (correct / total) * 100
            mean_optimal_tour_distance = optimal_tour_distances.sum().item() / len(self.test_dataset)
            mean_predicted_tour_distance = predicted_tour_distances.sum().item() / len(self.test_dataset)
            mean_opt_gap = (mean_predicted_tour_distance - mean_optimal_tour_distance) / mean_optimal_tour_distance * 100
            self.print(
                f"\ncorrect={correct}",
                f"\ntotal={total}",
                f"\nnode prediction(hit ratio) = {hit_ratio} %",
                f"\nmean_optimal_tour_distance = {mean_optimal_tour_distance}",
                f"\nmean_predicted_tour_distance = {mean_predicted_tour_distance}",
                f"\nmean_opt_gap = {mean_opt_gap}  %",
            )

if __name__ == "__main__":
    cfg = OmegaConf.create({
        "train_data_path": "./reordered(tour_only)_tsp20_train_concorde.txt",
        "val_data_path": "./reordered(tour_only)_tsp20_test_concorde.txt", # ./reordered_tsp20_train_concorde.txt
        "node_size": 20,
        "train_batch_size": 80,
        "val_batch_size": 80,
        "resume_checkpoint": "./logs/lightning_logs/version_6/checkpoints/TSP20-epoch=65-val_loss=25.9200.ckpt",
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
    
    # tsp_model = TSPModel(cfg)
    
    tsp_model = TSPModel.load_from_checkpoint(cfg.resume_checkpoint)
    tsp_model.set_cfg(cfg)
    
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