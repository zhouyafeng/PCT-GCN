
import torch
import torch.nn as nn


class GCNHead(nn.Module):

    def __init__(self, num_class: int, in_channels: int, dropout: float = 0):
        super().__init__()

        self.mode = "GCN"
        self.num_class = num_class
        self.in_channels = in_channels
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.fc = nn.Linear(in_channels, num_class)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(x))
