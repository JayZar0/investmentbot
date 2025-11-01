import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import pickle
import os

## This file will contain the functions that will be used to load the training
## dataset and the validation dataset

# This is the class that will initialize the dataset
class StockTradingDataset(Dataset):
    def __init__(self, csv_file, header=True, train=True, device='cpu', 
                 normalize_inputs=True, normalize_outputs=True, use_one_hot=True,
                 scaler_dir='scalers'):
        """
        Args:
            csv_file: Path to CSV file
            header: Whether CSV has header row
            train: Whether this is training data (affects train/test split)
            device: Device to load tensors on
            normalize_inputs: Whether to normalize input features
            normalize_outputs: Whether to normalize output targets
            use_one_hot: Whether to use one-hot encoding for stocks (vs integer)
            scaler_dir: Directory to save/load scalers for consistency
        """
        self.device = device
        self.train = train
        self.normalize_inputs = normalize_inputs
        self.normalize_outputs = normalize_outputs
        self.use_one_hot = use_one_hot
        self.scaler_dir = scaler_dir
        
        # Create scaler directory if it doesn't exist
        if not os.path.exists(scaler_dir):
            os.makedirs(scaler_dir)
        
        # Load data using pandas for better handling
        df_full = pd.read_csv(csv_file)
        
        # CRITICAL: Get ALL unique stocks from entire dataset BEFORE splitting
        # This ensures train and validation use the same one-hot encoding dimensions
        all_unique_stocks = df_full.iloc[:, 0].unique()
        num_unique_stocks = len(all_unique_stocks)
        
        # Create a fixed mapping of stock codes to indices (must be consistent across train/val)
        stock_to_idx = {stock: idx for idx, stock in enumerate(sorted(all_unique_stocks))}
        self.all_unique_stocks = sorted(all_unique_stocks)
        
        # Split train/validation (80/20)
        split_idx = int(len(df_full) * 0.8)
        if train:
            df = df_full.iloc[:split_idx].copy()
        else:
            df = df_full.iloc[split_idx:].copy()
        
        # Extract stock symbols
        self.stock_codes = df.iloc[:, 0].values  # Stock symbols
        
        if use_one_hot:
            # One-hot encoding for stocks using fixed mapping
            # Map each stock to its index, then create one-hot vectors
            code_indices = np.array([stock_to_idx[stock] for stock in self.stock_codes])
            stock_one_hot = np.zeros((len(code_indices), num_unique_stocks), dtype=np.float32)
            stock_one_hot[np.arange(len(code_indices)), code_indices] = 1.0
            self.stock_features = stock_one_hot
            
            # Save encoder for inference (only during training)
            if train:
                code_encoder = LabelEncoder()
                code_encoder.fit(self.all_unique_stocks)  # Fit on all stocks
                with open(os.path.join(scaler_dir, 'stock_encoder.pkl'), 'wb') as f:
                    pickle.dump(code_encoder, f)
                    pickle.dump(self.all_unique_stocks, f)
                    pickle.dump(stock_to_idx, f)
        else:
            # Simple integer encoding (normalized) - also use fixed mapping
            code_indices = np.array([stock_to_idx[stock] for stock in self.stock_codes], dtype=np.float32)
            # Normalize to [0, 1] range
            if train:
                # Use all possible indices (0 to num_unique_stocks-1) for normalization
                self.code_min, self.code_max = 0.0, float(num_unique_stocks - 1)
                code_labels = (code_indices - self.code_min) / (self.code_max - self.code_min + 1e-8)
                with open(os.path.join(scaler_dir, 'code_scaler.pkl'), 'wb') as f:
                    pickle.dump({'min': self.code_min, 'max': self.code_max, 'num_stocks': num_unique_stocks}, f)
            else:
                with open(os.path.join(scaler_dir, 'code_scaler.pkl'), 'rb') as f:
                    scaler_info = pickle.load(f)
                    self.code_min = scaler_info['min']
                    self.code_max = scaler_info['max']
                    code_labels = (code_indices - self.code_min) / (self.code_max - self.code_min + 1e-8)
            self.stock_features = code_labels.reshape(-1, 1)
        
        # Extract and engineer date features (better than raw timestamps)
        dates = pd.to_datetime(df.iloc[:, 1], format="%Y-%m-%d")
        date_features = np.zeros((len(dates), 6), dtype=np.float32)
        
        # Normalize year
        years = dates.dt.year.values.astype(np.float32)
        date_features[:, 1] = dates.dt.month.values.astype(np.float32) / 12.0  # Normalized to [0,1]
        date_features[:, 2] = dates.dt.day.values.astype(np.float32) / 31.0   # Normalized to [0,1]
        date_features[:, 3] = dates.dt.dayofweek.values.astype(np.float32) / 6.0  # Day of week [0,1]
        date_features[:, 4] = dates.dt.dayofyear.values.astype(np.float32) / 365.0  # Day of year [0,1]
        # Days since first date (normalized)
        days_since_start = (dates - dates.min()).dt.days.values.astype(np.float32)
        
        if train:
            self.year_min, self.year_max = years.min(), years.max()
            date_features[:, 0] = (years - self.year_min) / (self.year_max - self.year_min + 1e-8)
            self.date_max = days_since_start.max()
            date_features[:, 5] = days_since_start / (self.date_max + 1e-8)
            # Save date scaler
            with open(os.path.join(scaler_dir, 'date_scaler.pkl'), 'wb') as f:
                pickle.dump({
                    'max': self.date_max, 
                    'min_date': dates.min(), 
                    'year_min': self.year_min, 
                    'year_max': self.year_max
                }, f)
        else:
            with open(os.path.join(scaler_dir, 'date_scaler.pkl'), 'rb') as f:
                scaler_info = pickle.load(f)
                date_features[:, 0] = (years - scaler_info['year_min']) / (scaler_info['year_max'] - scaler_info['year_min'] + 1e-8)
                date_features[:, 5] = days_since_start / (scaler_info['max'] + 1e-8)
        
        # Extract outputs (open, high, low, close, volume)
        outputs = df.iloc[:, 2:7].values.astype(np.float32)
        
        # Normalize outputs (critical for model training)
        if normalize_outputs:
            if train:
                self.output_scaler = StandardScaler()
                outputs = self.output_scaler.fit_transform(outputs)
                with open(os.path.join(scaler_dir, 'output_scaler.pkl'), 'wb') as f:
                    pickle.dump(self.output_scaler, f)
            else:
                with open(os.path.join(scaler_dir, 'output_scaler.pkl'), 'rb') as f:
                    self.output_scaler = pickle.load(f)
                outputs = self.output_scaler.transform(outputs)
        
        # Combine features: stock features + date features
        if use_one_hot:
            self.inputs = np.hstack([self.stock_features, date_features]).astype(np.float32)
        else:
            self.inputs = np.hstack([self.stock_features, date_features]).astype(np.float32)
        
        # Normalize inputs if requested
        if normalize_inputs:
            if train:
                self.input_scaler = StandardScaler()
                self.inputs = self.input_scaler.fit_transform(self.inputs)
                with open(os.path.join(scaler_dir, 'input_scaler.pkl'), 'wb') as f:
                    pickle.dump(self.input_scaler, f)
            else:
                with open(os.path.join(scaler_dir, 'input_scaler.pkl'), 'rb') as f:
                    self.input_scaler = pickle.load(f)
                self.inputs = self.input_scaler.transform(self.inputs)
        
        self.outputs = outputs
        
        print(f"Dataset loaded: {len(self.inputs)} samples")
        print(f"Input shape: {self.inputs.shape}, Output shape: {self.outputs.shape}")
        if train:
            print(f"Input range: [{self.inputs.min():.4f}, {self.inputs.max():.4f}]")
            print(f"Output range: [{self.outputs.min():.4f}, {self.outputs.max():.4f}]")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.inputs[idx]).to(device=self.device)
        y = torch.from_numpy(self.outputs[idx]).to(device=self.device)
        return x, y
    
    def inverse_transform_outputs(self, outputs):
        """Convert normalized outputs back to original scale"""
        if hasattr(self, 'output_scaler') and self.output_scaler:
            if isinstance(outputs, torch.Tensor):
                outputs = outputs.cpu().numpy()
            return self.output_scaler.inverse_transform(outputs)
        return outputs
