import torch
from model import make_model, subsequent_mask

batch_size, node_size, node_dim = 4, 100, 2
src = torch.rand(batch_size, node_size, node_dim)

model = make_model(src_sz=node_size, tgt_sz=node_size, N=6)
model.eval()
memory = model.encode(src)
ys = torch.zeros(batch_size, 1).type(torch.long)

visited = torch.zeros(batch_size, node_size, dtype=torch.bool)
visited[:, 0] = True
for i in range(node_size - 1):
    # memory, tgt, tgt_mask
    tgt_mask = subsequent_mask(ys.size(1)).type(torch.long)
    out = model.decode(memory, src, ys, visited, tgt_mask)

    print("out", out[:, -1].shape)
    print("visited", visited.shape)
    print()
    
    prob = model.generator(out[:, -1], visited)
    _, next_word = torch.max(prob, dim=1)
    
    visited[torch.arange(batch_size), next_word] = True
    ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)

"""
for batch_idx, nodes in enumerate(ys):
    for node_idx, node in enumerate(nodes):
        print(f"for batch {batch_idx} node_idx, node: {node_idx}, {node}")
    print()
"""