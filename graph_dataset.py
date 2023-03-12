import os
import torch
from torch_geometric.data import Data, Dataset
from graph_data_preprocess import preprocess_graph_data
import multiprocessing as mp

class GraphDataset(Dataset):
    def __init__(self, root, model_name, bjorn=True, transform=None, pre_transform=None):
        self._indices = None
        self.transforms = transform
        self.root = root
        self.model_name = model_name
        self.model_name_list = ["hollow_1", "hollow_2", "hollow_3", "hollow_4", "hollow_5", "townhouse_2", "townhouse_3", "townhouse_5", "townhouse_6", "townhouse_7"]
        self.raw_data_dir = os.path.join(root, "bjorn", "small_10_base_20")
        self.bjorn = bjorn
        self.processed_data_dir = os.path.join(self.root, "ml_data", self.model_name)
        # self.preprocess_graph_data = preprocess_graph_data
        # super(GraphDataset, self).__init__(root, transform, pre_transform)
        # self.data, self.slices = torch.load(self.processed_paths[0])
        if not self.is_processed:
            self.process()
        # self.data_list = os.listdir(self.processed_data_dir)

    def len(self) -> int:
        return len(self.processed_file_names)

    @property
    def processed_file_names(self):
        return os.listdir(self.processed_data_dir)
    
    @property
    def is_processed(self):
        return os.path.isdir(self.processed_dir)
        # return True
    
    @property
    def raw_file_names(self):
        return os.listdir(self.raw_data_dir)

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # map preprocess_graph_data to all models in parallel
        num_cores = len(self.model_name_list)
        pool = mp.Pool(num_cores)
        pool.map(preprocess_graph_data, self.model_name_list)

        pool.close()
        pool.join()
        # preprocess_graph_data(self.root, self.model_name, self.bjorn)

    def __repr__(self):
        return '{}({})'.format(self.model_name, len(self))
    
    def transform(self, data):
        # normalize data using mean and std of x
        data.x[0] = (data.x[0] - data.x[0].mean()) / data.x[0].std()
        data.y = (data.y - data.x.mean()) / data.x.std()
        return data
    
    def get(self, idx: int):
        data = torch.load(os.path.join(self.processed_data_dir, self.processed_file_names[idx]))
        if self.transforms:
            data = self.transform(data)
        return data
