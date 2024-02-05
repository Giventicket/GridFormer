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
        
        self.tsp_instances = []
        self.tsp_tours = []
        self.reversed_tsp_tours = []

        self._readDataFile()
        
        # self.tsp_instances = self.tsp_instances[:64] # delete
        # self.tsp_tours = self.tsp_tours[:64] # delete
        
        self.raw_data_size = len(self.tsp_instances)
        self.max_node_size = len(self.tsp_tours[0])
        
        self.src = []
        self.tgt = []
        self.visited_mask = []
        self.tgt_y = []
        self.ntokens = []
        self._process()
        self.data_size = len(self.src)

        print()
        print("#### processing dataset... ####")
        print("data_path:", data_path)
        print("raw_data_size:", self.raw_data_size)
        print("max_node_size:", self.max_node_size)
        print("data_size:", self.data_size)
        print()

    def _readDataFile(self):
        """
        read validation dataset from "https://github.com/Spider-scnu/TSP"
        """
        with open(self.data_path, "r") as fp:
            tsp_set = fp.readlines()
            for tsp in tsp_set:
                tsp = tsp.split("output")
                tsp_instance = tsp[0].split()

                tsp_instance = [int(i) for i in tsp_instance]
                tsp_instance = torch.LongTensor(tsp_instance)
                self.tsp_instances.append(tsp_instance)

                tsp_tour = tsp[1].split()
                tsp_tour = [int(i) for i in tsp_tour]
                reversed_tsp_tour = tsp_tour[1:].copy()
                reversed_tsp_tour.reverse()
                
                tsp_tour = torch.LongTensor(tsp_tour[:-1])
                self.tsp_tours.append(tsp_tour)
                
                reversed_tsp_tour = torch.LongTensor(reversed_tsp_tour)
                self.reversed_tsp_tours.append(reversed_tsp_tour)
        return

    def _process(self):
        for tsp_instance, tsp_tour in tqdm(zip(self.tsp_instances, self.tsp_tours)):
            self.ntokens.append(torch.LongTensor([1]))
            self.src.append(tsp_instance)
            
            if self.use_start_token:
                self.tgt.append(torch.tensor([self.grid_size * self.grid_size], dtype = torch.long))
                mask = torch.ones(self.grid_size * self.grid_size + 1, dtype=torch.int8)
            else:
                self.tgt.append(torch.tensor([tsp_tour[0]], dtype = torch.long))
                mask = torch.ones(self.grid_size * self.grid_size, dtype=torch.int8)
            
            for v in range(len(tsp_tour)):
                mask[tsp_tour[v]] = mask[tsp_tour[v]] - 1 # unvisited
            
            if self.use_start_token:
                mask[self.grid_size * self.grid_size] = mask[self.grid_size * self.grid_size] + 1 # first visited
            else:
                mask[tsp_tour[0]] = mask[tsp_tour[0]] + 1 # first visited
            self.visited_mask.append(mask)
        return

    def __len__(self):
        return len(self.tsp_instances)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx], self.visited_mask[idx], self.ntokens[idx], self.tsp_tours[idx], self.reversed_tsp_tours[idx]
    
    def __iter__(self):
        for idx in range(len(self)):
            yield self.__getitem__(idx)

def make_tgt_mask(tgt):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != -1).unsqueeze(-2) # -1 equals blank
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
    return tgt_mask


def collate_fn(batch):
    src = [ele[0] for ele in batch]
    tgt = [ele[1] for ele in batch]
    visited_mask = [ele[2] for ele in batch]
    ntokens = [ele[3] for ele in batch]
    tsp_tours = [ele[4] for ele in batch]
    reversed_tsp_tours = [ele[5] for ele in batch]

    tgt = torch.stack(tgt, dim=0)
    
    return {
        "src": torch.stack(src, dim=0),
        "tgt": tgt,
        "visited_mask": torch.stack(visited_mask, dim=0),
        "ntokens": torch.stack(ntokens, dim=0),
        "tgt_mask": make_tgt_mask(tgt),
        "tsp_tours": torch.stack(tsp_tours, dim=0),
        "reversed_tsp_tours": torch.stack(reversed_tsp_tours, dim=0),
    }

if __name__ == "__main__":
    train_dataset = TSPDataset("./tsp20_grid_test.txt", grid_size = 100)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    for tsp_instances in tqdm(train_dataloader):
        print("tsp_instances")
        for k, v in tsp_instances.items():
            print(k, v.shape)
            if str(k) == "visited_mask":
                print(k, "sum", v.sum())
        print()
        break