import torch
import torch.nn as nn

class SimpleLossComputeWithMask:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, node_size):
        self.generator = generator
        self.criterion = criterion
        self.node_size = node_size

    def __call__(self, x, y_t, visited_mask, norm):
        x = self.generator(x, visited_mask)
        sloss = self.criterion(x.reshape(-1, self.node_size), y_t.reshape(-1), visited_mask.reshape(-1, self.node_size)) / norm
        return sloss

class LabelSmoothingWithMask(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, smoothing=0.0):
        super(LabelSmoothingWithMask, self).__init__()
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
        true_dist = torch.zeros_like(x, device = device) # [T, N]

        smoothing_dist = self.smoothing / (~visited_mask).sum(-1, keepdim = True) # [T, 1] 
        true_dist[~visited_mask] = smoothing_dist.repeat(1, node_size)[~visited_mask] # [T, N]
        indices = torch.arange(T, device = device) # [T]
        true_dist[indices, target] += self.confidence
        self.true_dist = true_dist

        return self.criterion(x, true_dist)
    
class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y_t, norm):
        x = self.generator(x)
        _, _, max_node_size = x.shape
        sloss = self.criterion(x.reshape(-1, max_node_size), y_t.contiguous().reshape(-1)) / norm
        return sloss
    
class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size

    def forward(self, x, target):
        T, N = x.shape
        device = x.device
        
        batch_indices = torch.arange(T, device = device, dtype=torch.long)
        node_indices = torch.arange(N, device = device, dtype=torch.long)
        
        true_dist = torch.zeros_like(x, device = device) # [T, N]
        true_dist[batch_indices.unsqueeze(-1).repeat(1, N).reshape(-1), node_indices.unsqueeze(0).repeat(T, 1).reshape(-1)] = self.smoothing / (N - 1) # [T, N]
        true_dist[batch_indices, target[batch_indices]] = self.confidence
        
        mask = torch.nonzero(target == -1)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        
        return self.criterion(x, true_dist)