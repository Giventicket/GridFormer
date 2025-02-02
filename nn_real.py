import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import tqdm
import math

class TSPDataset(Dataset):
    def __init__(
        self,
        data_path=None,
    ):
        super(TSPDataset, self).__init__()
        self.data_path = data_path
        self.tsp_instances = []
        self.tsp_tours = []

        self._readDataFile()

    def _readDataFile(self):
        """
        read validation dataset from "https://github.com/Spider-scnu/TSP"
        """
        with open(self.data_path, "r") as fp:
            tsp_set = fp.readlines()
            for tsp in tsp_set:
                tsp = tsp.split("output")
                tsp_instance = tsp[0].split()
                tsp_instance = [float(i) for i in tsp_instance]
                loc_x = torch.FloatTensor(tsp_instance[::2])
                loc_y = torch.FloatTensor(tsp_instance[1::2])
                tsp_instance = torch.stack([loc_x, loc_y], dim=1)
                self.tsp_instances.append(tsp_instance)

                tsp_tour = tsp[1].split()
                tsp_tour = [(int(i) - 1) for i in tsp_tour]
                tsp_tour = torch.LongTensor(tsp_tour[:-1])
                self.tsp_tours.append(tsp_tour)
        return

    def __len__(self):
        return len(self.tsp_instances)

    def __getitem__(self, idx):
        return self.tsp_instances[idx], self.get_tour_distance(self.tsp_instances[idx][self.tsp_tours[idx], :])
    
    def get_tour_distance(self, coordinates):
        rolled_coordinates = coordinates.roll(dims = 0, shifts = -1)
        segment_lengths = ((coordinates - rolled_coordinates) ** 2).sum(-1).sqrt() # [B, N]
        group_travel_distances = segment_lengths.sum(-1)
        return group_travel_distances
    
def nearest_neighbor(matrix, start_vertex=0):
    n = len(matrix)
    visited = [False] * n
    path = [start_vertex]
    total_distance = 0

    for _ in range(n):
        current_vertex = path[-1]
        min_distance = float('inf')
        nearest_vertex = None

        for neighbor in range(n):
            if not visited[neighbor] and matrix[current_vertex][neighbor] < min_distance:
                min_distance = matrix[current_vertex][neighbor]
                nearest_vertex = neighbor

        path.append(nearest_vertex)
        total_distance += min_distance
        visited[nearest_vertex] = True

    back_to_start = (matrix[nearest_vertex][start_vertex])

    return path[1:], total_distance+back_to_start

rawdata = TSPDataset("tsp20_real_test.txt")


pred = []
opt = []
gaps = []
for inst, opt_dist in rawdata:
    start_vertex = 0
    _, pred_dist = nearest_neighbor(torch.cdist(inst, inst), start_vertex)
    opt.append(opt_dist)
    pred.append(pred_dist)
    gap = (pred_dist.item() - opt_dist.item()) / opt_dist.item()
    gaps.append(gap)

pred = torch.tensor(pred)
opt =  torch.tensor(opt)
gaps =  torch.tensor(gaps)

gap = ((pred.sum() - opt.sum()) / opt.sum()) * 100
# print(gap.item(), "%")
print(gaps.mean().item() * 100, "%")