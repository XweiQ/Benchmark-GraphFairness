import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nlayers, nclass, dropout):
        super(GCN, self).__init__()
        self.body = GCN_Body(nfeat,nhid,nlayers,dropout)
        self.fc = nn.Linear(nhid, nclass)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        x = self.body(x, edge_index)
        x = self.fc(x)
        return x

class GCN_Body(nn.Module):
    def __init__(self, nfeat, nhid, nlayers, dropout):
        super(GCN_Body, self).__init__()

        self.gc_layer = nlayers
        self.gc1 = GCNConv(nfeat, nhid)
        self.hidden = nn.ModuleList()
        self.hidden.append(self.gc1)
        for _ in range(nlayers-1):
            self.hidden.append(GCNConv(nhid, nhid))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        for layer in range(self.gc_layer-1):
            x = F.relu(self.hidden[layer](x, edge_index))
            x = self.dropout(x)
        x = F.relu(self.hidden[-1](x, edge_index))
        return x    




