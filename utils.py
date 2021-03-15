from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import DataLoader

# df = pd.read_csv('df.csv')
# graph_df_r = pd.read_csv('graph_df_r.csv')


class COVIDDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(COVIDDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']
        # pass

    def download(self):
        pass

    def process(self):
        data_list = []
        x_list = []
        y_list = []
        for i in range(5, df.shape[1]):
            x_cut = df.iloc[:, i - 5:i - 1]
            y_cut = df.iloc[:, i]
            x_list = x_cut.values.tolist()
            y_list = y_cut.values.tolist()
            x = torch.FloatTensor(x_list)
            y = torch.FloatTensor(y_list)
            edge_list = graph_df_r.values.tolist()
            edge_index = torch.LongTensor(edge_list)

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def data_loader(dataset, batch_size):
    data_loader_ = DataLoader(dataset, batch_size, shuffle=True)
    return data_loader_


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_pred + y_true))) * 100
