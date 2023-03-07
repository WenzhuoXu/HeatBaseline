from graph_dataset import GraphDataset
from torch_geometric.data import DataLoader
import torch


if __name__ == '__main__':
    dataset = GraphDataset(root='d:/Work/research/data/hammer', model_name='hollow_1')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for data in loader:
        print(data)
        break

    print(data)