import torch
import torch.nn as nn

class SimpleLossComputeWithMask:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y_t, visited_mask, norm):
        x = self.generator(x, visited_mask)
        tgt_vocab = x.shape[-1]
        sloss = self.criterion(x.reshape(-1, tgt_vocab), y_t.reshape(-1), visited_mask.reshape(-1, tgt_vocab)) / norm
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
        T, grid_size = x.shape # [T, N]
        true_dist = torch.zeros_like(x, device = device) # [T, N]

        smoothing_dist = self.smoothing / (~visited_mask).sum(-1, keepdim = True) # [T, 1] 
        true_dist[~visited_mask] = smoothing_dist.repeat(1, grid_size)[~visited_mask] # [T, N]
        indices = torch.arange(T, device = device) # [T]
        true_dist[indices, target] += self.confidence
        self.true_dist = true_dist

        return self.criterion(x, true_dist)