import torch.nn as nn
from tqdm import tqdm

import torch
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, dropout, size = 50, num_classes=2, num_layer= 10):
        super(MLP, self).__init__()

        self.hidden = nn.ModuleList()
        for _ in range(num_layer-2):
            self.hidden.append(nn.Linear(size, size))
        self.first = nn.Linear(input_size, size)
        self.last = nn.Linear(size, num_classes)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.first(x))
        x = self.dropout(x)
        for layer in self.hidden:
            x = F.relu(layer(x))
            x = self.dropout(x)

        x = self.last(x)
        return x
