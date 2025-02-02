import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

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

        self._readDataFile()
        
        # self.tsp_instances = self.tsp_instances[:10] # delete
        # self.tsp_tours = self.tsp_tours[:10] # delete
        
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
                if self.use_start_token:
                    tsp_tour = [self.grid_size * self.grid_size] + [int(i) for i in tsp_tour]
                else:
                    tsp_tour = [int(i) for i in tsp_tour]
                tsp_tour = torch.LongTensor(tsp_tour[:-1])
                self.tsp_tours.append(tsp_tour)
        return

    def _process(self):
        N = len(self.tsp_instances[0])
        for tsp_instance, tsp_tour in tqdm(zip(self.tsp_instances, self.tsp_tours)):
            
            if self.use_start_token:
                ntoken = len(tsp_tour)
            else:
                ntoken = len(tsp_tour) - 1
                
            self.ntokens.append(torch.LongTensor([ntoken]))
            self.src.append(tsp_instance)
            self.tgt.append(tsp_tour[0:ntoken])
            self.tgt_y.append(tsp_tour[1 : ntoken + 1])
            
            if self.use_start_token:
                mask = torch.ones(ntoken, self.grid_size * self.grid_size + 1, dtype=torch.int8)
            else:
                mask = torch.ones(ntoken, self.grid_size * self.grid_size, dtype=torch.int8)
            
            for v in range(ntoken):
                mask[: , self.tgt[-1][v]] = mask[: , self.tgt[-1][v]] - 1 # unvisited
            
            for v in range(ntoken):
                mask[v: , self.tgt[-1][v]] = mask[v: , self.tgt[-1][v]] + 1 # visited
            self.visited_mask.append(mask)
            
        return

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx], self.visited_mask[idx], self.tgt_y[idx], self.ntokens[idx]
        
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
    visited_mask = pad_sequence([ele[2] for ele in batch], batch_first=True, padding_value=False)
    tgt_y = [ele[3] for ele in batch]
    ntokens = [ele[4] for ele in batch]
    
    N = max(ntokens)

    tgt = [torch.cat([ele, torch.LongTensor([-1] * (N - ele.shape[0]))], dim=-1) for ele in tgt]
    tgt_y = [torch.cat([ele, torch.LongTensor([-1] * (N - ele.shape[0]))], dim=-1) for ele in tgt_y]

    tgt = torch.stack(tgt, dim=0)
    
    return {
        "src": torch.stack(src, dim=0),
        "tgt": tgt,
        "visited_mask": visited_mask,
        "tgt_y": torch.stack(tgt_y, dim=0),
        "ntokens": torch.stack(ntokens, dim=0),
        "tgt_mask": make_tgt_mask(tgt),
    }

if __name__ == "__main__":
    train_dataset = TSPDataset("./tsp20_grid100_train.txt", grid_size = 100, use_start_token=False)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    for tsp_instances in tqdm(train_dataloader):
        print("tsp_instances")
        for k, v in tsp_instances.items():
            print(k, v.shape)
        print()
        break