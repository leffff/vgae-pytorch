from typing import Tuple

import torch
from torch import nn
from torch_geometric.data import Data


class Encoder(nn.Module):
    def __init__(self, hidden_model, mean_layers, std_model):
        super().__init__()

        self.hidden_model = hidden_model
        self.mean_layers = mean_layers
        self.std_model = std_model

    def encode(self,
               x: torch.Tensor,
               edge_index: torch.LongTensor,
               edge_attr: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:

        if isinstance(edge_attr, type(None)):
            hidden = self.hidden_model(x, edge_index, edge_attr)
            mean = self.mean_model(hidden, edge_index, edge_attr)
            std = self.std_model(hidden, edge_index, edge_attr)
        else:
            hidden = self.hidden_model(x, edge_index)
            mean = self.mean_model(hidden, edge_index)
            std = self.std_model(hidden, edge_index)

        return mean, std

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        mu, logvar = self.encode(x, edge_index, edge_attr)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, activation=torch.sigmoid, dropout: float = 0.1):
        super().__init__()

        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.dropout(z)
        adj_reconstructed = self.activation(torch.bmm(z, z.T))
        return adj_reconstructed


class VGAE(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def reparametrize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std) + mu
        else:
            return mu

    def forward(self, data: Data):
        mu, logvar = self.encoder(data)
        z = self.reparametrize(mu, logvar)
        adj = self.decoder(z)
        return adj, mu, logvar
