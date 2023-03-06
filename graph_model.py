import torch
import torch_geometric.nn as pyg_nn


class GraphNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(GraphNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(pyg_nn.GCNConv(input_dim, hidden_dim))
        for i in range(num_layers - 2):
            self.convs.append(pyg_nn.GCNConv(hidden_dim, hidden_dim))
        self.convs.append(pyg_nn.GCNConv(hidden_dim, output_dim))

        self.bns = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = torch.relu(x)
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return torch.sigmoid(x)