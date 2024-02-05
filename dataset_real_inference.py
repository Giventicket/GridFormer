import torch
from torch.utils.data import DataLoader, Dataset
from model import subsequent_mask

from tqdm import tqdm
from pprint import pprint

class TSPDataset(Dataset):
    def __init__(
        self,
        data_path,
        grid_size,
        use_start_token
    ):
        super(TSPDataset, self).__init__()
        self.data_path = data_path
        self.grid_size = grid_size
        self.use_start_token = use_start_token
        
        self.real_tsp_instances = []
        self.real_tsp_tours = []
        self.real_reversed_tsp_tours = []
        
        self.grid_tsp_instances = []
        self.grid_tsp_tours = []
        self.grid_reversed_tsp_tours = []
        
        self._readDataFile()
        
        self.raw_data_size = len(self.real_tsp_instances)
        self.max_node_size = len(self.real_tsp_tours[0])
        
        self.src = []
        self.tgt = []
        self.visited_mask = []
        self.tgt_y = []
        self.ntokens = []
        self.real_src = []
        
        self._process()
        self.data_size = len(self.src)

        print()
        print("#### processing dataset... ####")
        print("data_path:", data_path)
        print("raw_data_size:", self.raw_data_size)
        print("max_node_size:", self.max_node_size)
        print("data_size:", self.data_size)
        print()
    
    def _point2grid(self, nodes_coord):
        nodes_coord_scaled = (nodes_coord * self.grid_size).to(torch.long)
        grid_indices = nodes_coord_scaled[:, 0] + nodes_coord_scaled[:, 1] * self.grid_size
        return grid_indices

    def _readDataFile(self):
        """
        read validation dataset from "https://github.com/Spider-scnu/TSP"
        """
        with open(self.data_path, "r") as fp:
            tsp_set = fp.readlines()
            for tsp in tsp_set:
                tsp = tsp.split("output")
                tsp_instance = tsp[0].split()

                real_tsp_instance = [float(i) for i in tsp_instance]
                loc_x = torch.FloatTensor(real_tsp_instance[::2])
                loc_y = torch.FloatTensor(real_tsp_instance[1::2])
                real_tsp_instance = torch.stack([loc_x, loc_y], dim=1)
                self.real_tsp_instances.append(real_tsp_instance)

                grid_tsp_instance = self._point2grid(real_tsp_instance)
                self.grid_tsp_instances.append(grid_tsp_instance)

                tsp_tour = tsp[1].split()
                tsp_tour = [int(i) - 1 for i in tsp_tour]
                reversed_tsp_tour = tsp_tour[1:].copy()
                reversed_tsp_tour.reverse()
                
                tsp_tour = torch.LongTensor(tsp_tour[:-1])
                reversed_tsp_tour = torch.LongTensor(reversed_tsp_tour)
                
                real_tsp_tour = real_tsp_instance[tsp_tour, :]
                real_reversed_tsp_tour = real_tsp_instance[reversed_tsp_tour, :]
                
                self.real_tsp_tours.append(real_tsp_tour)
                self.real_reversed_tsp_tours.append(real_reversed_tsp_tour)
                
                grid_tsp_tour = self._point2grid(real_tsp_tour)
                grid_reversed_tsp_tour = self._point2grid(real_reversed_tsp_tour)
                
                self.grid_tsp_tours.append(grid_tsp_tour)
                self.grid_reversed_tsp_tours.append(grid_reversed_tsp_tour)
        return

    def _process(self):
        for grid_tsp_instance, grid_tsp_tour, real_tsp_instance in tqdm(zip(
                self.grid_tsp_instances, 
                self.grid_tsp_tours,
                self.real_tsp_instances, 
                )):
            self.real_src.append(real_tsp_instance)
            self.ntokens.append(torch.LongTensor([1]))
            self.src.append(grid_tsp_instance)
            
            if self.use_start_token:
                self.tgt.append(torch.tensor([self.grid_size * self.grid_size], dtype = torch.long))
                mask = torch.ones(self.grid_size * self.grid_size + 1, dtype = torch.int8)
            else:
                self.tgt.append(torch.tensor([grid_tsp_tour[0]], dtype = torch.long))
                mask = torch.ones(self.grid_size * self.grid_size, dtype = torch.int8)
            
            for v in range(len(grid_tsp_tour)):
                mask[grid_tsp_tour[v]] = mask[grid_tsp_tour[v]] - 1 # unvisited
            
            if self.use_start_token:
                mask[self.grid_size * self.grid_size] = mask[self.grid_size * self.grid_size] + 1 # first visited
            else:
                mask[grid_tsp_tour[0]] = mask[grid_tsp_tour[0]] + 1 # first visited
            self.visited_mask.append(mask)
        return

    def __len__(self):
        return len(self.real_tsp_instances)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx], self.visited_mask[idx], self.ntokens[idx], self.grid_tsp_tours[idx], self.grid_reversed_tsp_tours[idx], self.real_tsp_tours[idx], self.real_reversed_tsp_tours[idx], self.real_src[idx]
    
    def __iter__(self):
        for idx in range(len(self)):
            yield self.__getitem__(idx)

def make_tgt_mask(tgt):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != -1).unsqueeze(-2) # -1 equals blank
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
    return tgt_mask


def collate_fn(batch):
    grid_src = [ele[0] for ele in batch]
    tgt = [ele[1] for ele in batch]
    visited_mask = [ele[2] for ele in batch]
    ntokens = [ele[3] for ele in batch]
    grid_tsp_tours = [ele[4] for ele in batch]
    grid_reversed_tsp_tours = [ele[5] for ele in batch]
    real_tsp_tours = [ele[6] for ele in batch]
    real_reversed_tsp_tours = [ele[7] for ele in batch]
    real_src = [ele[8] for ele in batch]

    tgt = torch.stack(tgt, dim=0)
    
    return {
        "grid_src": torch.stack(grid_src, dim=0),
        "tgt": tgt,
        "visited_mask": torch.stack(visited_mask, dim=0),
        "ntokens": torch.stack(ntokens, dim=0),
        "tgt_mask": make_tgt_mask(tgt),
        "grid_tsp_tours": torch.stack(grid_tsp_tours, dim=0),
        "grid_reversed_tsp_tours": torch.stack(grid_reversed_tsp_tours, dim=0),
        "real_tsp_tours": torch.stack(real_tsp_tours, dim=0),
        "real_reversed_tsp_tours": torch.stack(real_reversed_tsp_tours, dim=0),
        "real_src": torch.stack(real_src, dim=0),
    }

if __name__ == "__main__":
    train_dataset = TSPDataset("./tsp20_real_test.txt", grid_size = 100, use_start_token = False)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    for tsp_instances in tqdm(train_dataloader):
        print("tsp_instances")
        for k, v in tsp_instances.items():
            # print(k, v.shape)
            print(k, v)
            print()
        print()
        break