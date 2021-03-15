import os
import pandas as pd
import torch
device = torch.device('cuda')



class Config:
    directories = {"source data": "source_data",
                   "datasets": "datasets",
                   "graph visualizations": "graph_visualizations",
                   "training visualizations": "training_visualizations"}

    # make sure paths are created when the experiment is started
    for d in directories.values():
        try:
            os.makedirs(d)
        except FileExistsError:
            pass

    def __init__(self, hidden_channels: int = 16, batch_size: int = 8, node_embedding_method: str = "NodeSketch"):

        self.batch_size_list = [32]

        self.hidden_channel_list = [32, 32]

        self.batch_size = batch_size

        self.node_embedding_parameters = dict(compute_node_embeddings=True,
                                              embedding_method=node_embedding_method,
                                              merge_features=True)

        # parameters of training. Number of classes must match the selected dataset!
        self.training_parameters = {"hidden_channels": hidden_channels,
                                    "lr": 1e-4,     # for mutag: 1e-2, for hcp_17_51 1e-4
                                    "epochs": 500,
                                    "min_lr": 1e-6,     # for mutag: 1e-4, for hcp_17_51 1e-6
                                    "patience": 20,
                                    "threshold": 1e-6,
                                    "samples_for_final_test": 0.20,   # fraction of test split
                                    }
