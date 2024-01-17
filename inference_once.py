import torch
from dataset import TSPDataset, collate_fn
from torch.utils.data import DataLoader
from model import make_model, subsequent_mask

batch_size, node_size, node_dim = 4, 20, 2

train_dataset = TSPDataset(f"./tsp{node_size}_test_concorde.txt")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
model = make_model(src_sz=node_size, tgt_sz=node_size, N=6)
model.eval()

for batch in train_dataloader:
    src, tgt, visited_mask, tgt_y, ntokens, tgt_mask = batch["src"], batch["tgt"], batch["visited_mask"], batch["tgt_y"], batch["ntokens"], batch["tgt_mask"]
    memory = model.encode(src)
    print("memory", memory.shape)
    out = model.decode(memory, src, tgt, visited_mask, tgt_mask)
    
    """
    B, V, E = out.shape
    out_ls = []
    for b in range(B):
            out_ls.append(out[b, ntokens[b] - 1, :].squeeze(0))
    out = torch.stack(out_ls, dim = 0)
    print("out", out.shape)
    """

    B, V, E = out.shape
    gather_indices = (ntokens - 1).unsqueeze(-1)
    gather_indices = gather_indices.expand(-1, -1, E)
    out = torch.gather(out, 1, gather_indices).squeeze(1)

    prob = model.generator(out, visited_mask)
    _, next_word = torch.max(prob, dim=1)
    
    print(next_word)


    break
