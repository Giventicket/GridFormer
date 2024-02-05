import argparse

import time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from omegaconf import OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader, Dataset

from dataset_real_inference import TSPDataset, collate_fn, make_tgt_mask
from model import make_model, subsequent_mask
from loss import SimpleLossComputeWithMask, LabelSmoothingWithMask

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
        self.test_corrects = []
        self.test_optimal_grid_tour_distances = []
        self.test_predicted_grid_tour_distances = []
        self.test_optimal_real_tour_distances = []
        self.test_predicted_real_tour_distances = []
        
    def set_cfg(self, cfg):
        self.cfg = cfg
        self.save_hyperparameters(cfg)  # save config file with pytorch lightening

    def test_dataloader(self):
        self.test_dataset = TSPDataset(self.cfg.test_data_path, self.cfg.grid_size, cfg.use_start_token)
        test_dataloader = DataLoader(
            self.test_dataset, 
            batch_size = self.cfg.test_batch_size, 
            shuffle = False, 
            collate_fn = collate_fn,
            pin_memory=True
        )
        return test_dataloader

    def test_step(self, batch, batch_idx):
        grid_src = batch["grid_src"]
        tgt = batch["tgt"]
        visited_mask = batch["visited_mask"]
        ntokens = batch["ntokens"]
        tgt_mask = batch["tgt_mask"]
        grid_tsp_tours = batch["grid_tsp_tours"]
        grid_reversed_tsp_tours = batch["grid_reversed_tsp_tours"]
        real_tsp_tours = batch["real_tsp_tours"]
        real_reversed_tsp_tours = batch["real_reversed_tsp_tours"]
        real_src = batch["real_src"]
        
        batch_size = grid_tsp_tours.shape[0]
        decode_step = self.cfg.node_size if cfg.use_start_token else self.cfg.node_size - 1
        
        self.model.eval()
        with torch.no_grad():
            memory = self.model.encode(grid_src)
            grid_ys = tgt.clone()
            visited_mask = visited_mask.clone()
            for i in range(decode_step):
                # memory, tgt, tgt_mask
                tgt_mask = subsequent_mask(grid_ys.size(1)).type(torch.bool).to(grid_src.device)
                out = self.model.decode(memory, grid_ys, tgt_mask)
                prob = self.model.generator(out[:, -1], visited_mask > 0)
                _, next_word = torch.max(prob, dim=1)
                
                visited_mask[torch.arange(batch_size), next_word] = visited_mask[torch.arange(batch_size), next_word] + 1
                grid_ys = torch.cat([grid_ys, next_word.unsqueeze(-1)], dim=1)

            if cfg.use_start_token:
                grid_ys = grid_ys[:, 1:]
        
        correct = []
        for idx in range(batch_size):
            rolled_grid_ys = grid_ys[idx].roll(shifts = -1)
            rolled_tour = grid_reversed_tsp_tours[idx].roll(shifts = -1)
            edges_pred = set(tuple(sorted((grid_ys[idx][i].item(), rolled_grid_ys[i].item()))) for i in range(len(rolled_grid_ys)))
            edges_tour = set(tuple(sorted((grid_reversed_tsp_tours[idx][i].item(), rolled_tour[i].item()))) for i in range(len(rolled_tour)))
            counter_pred = Counter(edges_pred)
            counter_tour = Counter(edges_tour)

            common_edges = counter_pred & counter_tour
            correct.append(sum(common_edges.values()))
        
        correct = torch.tensor(correct)
        grid_tour_coord = self.grid_idx_to_coord(grid_tsp_tours)
        optimal_grid_tour_distance = self.get_tour_distance(grid_tour_coord)
        optimal_real_tour_distance = self.get_tour_distance(real_tsp_tours)
        predicted_grid_tour_distance = self.get_tour_distance(self.grid_idx_to_coord(grid_ys))
        
        real_ys = torch.zeros_like(real_src)
        for batch_idx in range(batch_size):
            _, indices_src = torch.sort(grid_src[batch_idx], dim = -1)
            _, indices_ys = torch.sort(grid_ys[batch_idx], dim = -1)
            real_ys[batch_idx, indices_ys] = real_src[batch_idx, indices_src]
        predicted_real_tour_distance = self.get_tour_distance(real_ys)
        
        result = {
            "correct": correct.tolist(),
            "optimal_grid_tour_distance": optimal_grid_tour_distance.tolist(),
            "optimal_real_tour_distance": optimal_real_tour_distance.tolist(),
            "predicted_grid_tour_distance": predicted_grid_tour_distance.tolist(),
            "predicted_real_tour_distance": predicted_real_tour_distance.tolist(),
            }    
        
        self.test_corrects.extend(result["correct"])
        self.test_optimal_grid_tour_distances.extend(result["optimal_grid_tour_distance"])
        self.test_predicted_grid_tour_distances.extend(result["predicted_grid_tour_distance"])
        self.test_optimal_real_tour_distances.extend(result["optimal_real_tour_distance"])
        self.test_predicted_real_tour_distances.extend(result["predicted_real_tour_distance"])
        
        """
        if self.trainer.is_global_zero:
            for idx in range(batch_size):
                print()
                print("predicted grid tour: ", grid_ys[idx].tolist())
                print("optimal grid tour: ", grid_tsp_tours[idx].tolist())
                print("opt, pred grid tour distance: ", optimal_grid_tour_distance[idx].item(), predicted_grid_tour_distance[idx].item())
                print("grid optimality gap: ", ((predicted_grid_tour_distance[idx].item() - optimal_grid_tour_distance[idx].item()) / optimal_grid_tour_distance[idx].item()) * 100, "%")
                print("opt, pred real tour distance: ", optimal_real_tour_distance[idx].item(), predicted_real_tour_distance[idx].item())
                print("real optimality gap: ", ((predicted_real_tour_distance[idx].item() - optimal_real_tour_distance[idx].item()) / optimal_real_tour_distance[idx].item()) * 100, "%")
                print("node prediction [hit ratio]: ", (correct[idx].item() / self.cfg.node_size) * 100 , "%")
                print()
        """
        
        return result
    
    def grid_idx_to_coord(self, seq):
        x_coordinates = (seq % self.cfg.grid_size) / self.cfg.grid_size  + 1 / (2 * self.cfg.grid_size)
        y_coordinates = (seq // self.cfg.grid_size) / self.cfg.grid_size  + 1 / (2 * self.cfg.grid_size)
        coordinates = torch.stack([x_coordinates, y_coordinates], dim=-1)
        return coordinates
    
    def get_tour_distance(self, coordinates):
        rolled_coordinates = coordinates.roll(dims = 1, shifts = -1)
        segment_lengths = ((coordinates - rolled_coordinates) ** 2).sum(-1).sqrt() # [B, N]
        group_travel_distances = segment_lengths.sum(-1)
        return group_travel_distances
    
    def on_test_epoch_end(self):
        corrects = self.all_gather(self.test_corrects)
        optimal_grid_tour_distances = self.all_gather(self.test_optimal_grid_tour_distances)
        predicted_grid_tour_distances = self.all_gather(self.test_predicted_grid_tour_distances)
        optimal_real_tour_distances = self.all_gather(self.test_optimal_real_tour_distances)
        predicted_real_tour_distances = self.all_gather(self.test_predicted_real_tour_distances)
        
        self.test_corrects.clear()
        self.test_optimal_grid_tour_distances.clear()
        self.test_predicted_grid_tour_distances.clear()
        self.test_optimal_real_tour_distances.clear()
        self.test_predicted_real_tour_distances.clear()
        
        if self.trainer.is_global_zero:
            corrects = torch.stack(corrects)
            optimal_grid_tour_distances = torch.stack(optimal_grid_tour_distances)
            predicted_grid_tour_distances = torch.stack(predicted_grid_tour_distances)
            optimal_real_tour_distances = torch.stack(optimal_real_tour_distances)
            predicted_real_tour_distances = torch.stack(predicted_real_tour_distances)
            
            correct = corrects.sum().item()
            total = self.cfg.node_size * len(self.test_dataset)
            hit_ratio = (correct / total) * 100
            
            grid_opt_gaps = (predicted_grid_tour_distances - optimal_grid_tour_distances) / optimal_grid_tour_distances
            mean_grid_opt_gap = grid_opt_gaps.mean().item() * 100
            
            real_opt_gaps = (predicted_real_tour_distances - optimal_real_tour_distances) / optimal_real_tour_distances
            mean_real_opt_gap = real_opt_gaps.mean().item() * 100
            
            self.print(
                f"\ncorrect={correct}",
                f"\ntotal={total}",
                f"\nnode prediction(hit ratio) = {hit_ratio} %",
                f"\nmean_grid_opt_gap = {mean_grid_opt_gap}  %",
                f"\nmean_real_opt_gap = {mean_real_opt_gap}  %",
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
    node_size = 20
    args = parse_arguments()
    cfg = OmegaConf.load(args.config)
    
    if not check_cfg(cfg):
        raise ValueError("Check config again!")
    
    pl.seed_everything(cfg.seed)
    
    tsp_model = TSPModel(cfg)
    
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