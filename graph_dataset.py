import os
import torch
from torch_geometric.data import Data, Dataset
from graph_data_preprocess import preprocess_graph_data
import multiprocessing as mp

class GraphDataset(Dataset):
    def __init__(self, root, model_name, bjorn=True, transform=None, pre_transform=None):
        self.model_name = model_name
        self.bjorn = bjorn
        # self.preprocess_graph_data = preprocess_graph_data
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        # self.data, self.slices = torch.load(self.processed_paths[0])
        self.processed_dir = os.path.join(self.root, "ml_data", self.model_name)

    @property
    def processed_file_names(self):
        return os.listdir(self.processed_dir)
    
    @property
    def is_processed(self):
        return os.path.isdir(self.processed_dir)

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        preprocess_graph_data(self.root, self.model_name, self.bjorn)

    def __repr__(self):
        return '{}({})'.format(self.model_name, len(self))
    
    def get(self, idx: int):
        data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))
        return data
