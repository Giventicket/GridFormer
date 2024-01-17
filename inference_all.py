import torch
from model import make_model, subsequent_mask

batch_size, node_size, node_dim = 1, 50, 2
data_path = f"./tsp{node_size}_test_concorde.txt"
tsp_instances = []
tsp_tours = []
with open(data_path, "r") as fp:
    tsp_set = fp.readlines()
    for tsp in tsp_set:
        tsp = tsp.split("output")
        tsp_instance = tsp[0].split()

        tsp_instance = [float(i) for i in tsp_instance]
        loc_x = torch.FloatTensor(tsp_instance[::2])
        loc_y = torch.FloatTensor(tsp_instance[1::2])
        tsp_instance = torch.stack([loc_x, loc_y], dim=1)
        tsp_instances.append(tsp_instance)

        tsp_tour = tsp[1].split()
        tsp_tour = [(int(i) - 1) for i in tsp_tour]
        tsp_tour = torch.LongTensor(tsp_tour[:-1])
        tsp_tours.append(tsp_tour)

for src, optimal_tour in zip(tsp_instances, tsp_tours):
    src = src.unsqueeze(0)
    optimal_tour = optimal_tour.unsqueeze(0)
    
    model = make_model(src_sz=node_size, tgt_sz=node_size, N=6)
    model.eval()
    memory = model.encode(src)
    ys = torch.zeros(batch_size, 1).type(torch.long)

    visited = torch.zeros(batch_size, node_size, dtype=torch.bool)
    visited[:, 0] = True
    for i in range(node_size - 1):
        # memory, tgt, tgt_mask
        tgt_mask = subsequent_mask(ys.size(1)).type(torch.bool)
        out = model.decode(memory, src, ys, visited, tgt_mask)
        prob = model.generator(out[:, -1], visited)
        _, next_word = torch.max(prob, dim=1)
        
        visited[torch.arange(batch_size), next_word] = True
        ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
        
    # print(ys)
    # print()
    # print(optimal_tour)
    # print()
    print(ys == optimal_tour)
    print()