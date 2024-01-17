import torch
import torch.nn as nn

class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, node_size):
        self.generator = generator
        self.criterion = criterion
        self.node_size = node_size

    def __call__(self, x, y, norm):
        device = x.device
        visited_mask = torch.zeros(x.shape[0], x.shape[1], self.node_size, dtype=torch.bool, device = device) # [B, V, N]
        visited_mask[:, :, 0] = True # every instance has visited first node
        valid_mask = (y != -1) # [B, V]
        B, V = valid_mask.shape
        batch_indices = torch.arange(B, device = device).unsqueeze(-1).expand_as(y) # [B, V]
        sequence_indices = torch.arange(V, device = device).unsqueeze(0).expand_as(y) # [B, V]
        batch_indices_valid = batch_indices[valid_mask]
        sequence_indices_valid = sequence_indices[valid_mask]
        y_valid = y[valid_mask]
        visited_mask[batch_indices_valid, sequence_indices_valid, y_valid] = True
        
        x = self.generator(x, visited_mask)
        # x = self.generator(x)
        
        sloss = self.criterion(x.reshape(-1, self.node_size), y.reshape(-1), visited_mask.reshape(-1, self.node_size)) / norm
        return sloss * norm, sloss

class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target, visited_mask):
        # visited_mask [T, N]
        assert x.size(1) == self.size
        device = x.device
        T, node_size = x.shape # [T, N]
        true_dist = torch.ones_like(x, device = device) # [T, N]
        
        smoothing_dist = self.smoothing / (~visited_mask).sum(-1, keepdim = True) # [T, 1] 
        true_dist[~visited_mask] = smoothing_dist.repeat(1, node_size)[~visited_mask] # [T, N]
        indices = torch.arange(T, device = device) # [T]
        true_dist[indices, target] += self.confidence
        self.true_dist = true_dist
        return self.criterion(x, true_dist)