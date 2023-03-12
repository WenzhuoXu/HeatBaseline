import torch
import torch_geometric.nn as pyg_nn
# import multiprocessing as mp


class GraphNet(torch.nn.Module):
    # Graph Neural Network.
    # Network processes nodal and elemental data separatly with linear layers before concatenating them.
    # The network then processes the concatenated data with a series of graph convolutional layers.
    def __init__(self, input_dim_node, input_dim_element, hidden_dim, output_dim, num_layers, dropout):
        super(GraphNet, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.node_feature_process = torch.nn.Linear(input_dim_node, hidden_dim)
        self.element_feature_process = torch.nn.Linear(input_dim_element, hidden_dim)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = pyg_nn.GENConv(hidden_dim, hidden_dim)
            norm = pyg_nn.LayerNorm(hidden_dim)
            act = torch.nn.LeakyReLU()

            layer = pyg_nn.DeepGCNLayer(conv, norm, act)
            self.convs.append(layer)

        self.output_layer = pyg_nn.GENConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, pos = data.x, data.edge_index.long(), data.pos
        x_node = self.node_feature_process(x)
        x_element = self.element_feature_process(pos)
        x = torch.add(x_node, x_element)
        for conv in self.convs:
            x = conv(x.float(), edge_index)
        x = self.output_layer(x, edge_index)
        return x
        