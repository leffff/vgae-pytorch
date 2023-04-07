import torch


def adj_matrix_from_edge_index(x, edge_index: torch.LongTensor) -> torch.Tensor:
    adj = torch.zeros((x.shape[0], x.shape[0]))
    adj[edge_index[0], edge_index[1]] = 1

    return adj