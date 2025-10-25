import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

## This file will contain the functions that will be used to load the training
## dataset and the validation dataset

# This is the class that will initialize the dataset
class StockTradingDataset(Dataset):
    def __init__(self, csv_file, header=True, train=True):
        data = np.loadtxt(csv_file, delimiter=',', dtype=str, skiprows=1 if header else 0)
        # If we're training we'll use the first 80% of the data:
        if train:
            data = data[0:int(len(data) * 0.8)]
        else:
            data = data[int(len(data) * 0.8):]

        # Encode stock codes as integers
        codes, code_labels = np.unique(data[:, 0], return_inverse=True)

        # Convert dates to integer timestamps
        dates_int = pd.to_datetime(data[:, 1], format="%Y-%m-%d").astype(np.int64)

        # Convert outputs to float
        outputs = data[:, 2:7].astype(np.float32)

        # Combine stock code ID + timestamp as inputs
        self.inputs = np.column_stack((code_labels, dates_int)).astype(np.float32)
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.inputs[idx])      # shape (2,)
        y = torch.from_numpy(self.outputs[idx])     # shape (5,)
        return x, y
