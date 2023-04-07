import torch
from torch import nn


class VGAELoss(nn.Module):
    def __init__(self, weight=None, norm: float = 0.5):
        super().__init__()

        self.weight = weight
        self.norm = norm

        self.ce = nn.CrossEntropyLoss(weight=self.weight)
        self.kl = nn.KLDivLoss()

    def forward(self, adj_output, mean, log_std, adj_target):
        cross_entropy = self.norm * self.ce(adj_output.flatten(), adj_target.flatten())
        kl_divergence = 0.5 / adj_output.size(0) * (1 + 2 * log_std - mean ** 2 - torch.exp(log_std) ** 2).sum(1).mean()
        loss = cross_entropy - kl_divergence
        return loss