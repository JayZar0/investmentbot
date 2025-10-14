import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

## This file will contain the functions that will be used to load the training
## dataset and the validation dataset

# This is the class that will initialize the dataset
class StockTradingDataset(Dataset):
    def __init__(self, csv_file, header=True, train=True):
        data = np.loadtxt(csv_file, delimiter=',', dtype=str)
        if header:
            data = np.delete(data, 0, axis=0)
        # If we're training we'll use the first 80% of the data:
        if train:
            data = data[0:int(len(data) * 0.8)]
        else:
            data = data[int(len(data) * 0.8):]

        # We'll store the data and the labels separately, since one of them
        # will start off as strings and we'll
        self.data = data[:, 1:6].astype(np.float32)
        self.classes, self.label = np.unique(data[:, 0], return_inverse=True)

        # need to convert dates into a string format so that I can perform analysis
        # based on the price of that stock during that day.

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item_data = self.data[idx, 1:6]
        item_label = self.label[idx]
        return torch.from_numpy(item_data), item_label
