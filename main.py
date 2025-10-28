from dataset import StockTradingDataset
from train import train_llm, validate_training
from neuralnetwork import MLP
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader

## This file will be used to contain the main code that run all the functions
## that will train the llm

# this will check if the device has a valid gpu for the training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')

# Grab the data from the csv files
print('Gathering data from csv')
training_data = StockTradingDataset('datasets/stock prices modified.csv', train=True, device=device)
validation_data = StockTradingDataset('datasets/stock prices modified.csv', train=False, device=device)

# create a predetermined batch size for the training dataset
batch_size = 8

# create the dataloader for training the llm
training_dataloader = DataLoader(training_data, batch_size=batch_size)
validation_dataloader = DataLoader(validation_data, batch_size=batch_size)

# set the layers size for the neural network out here rather than in the
# file/function itself so that everything can be modified in main
input_size = 2
hidden_size = 256
output_size = 5

# Create model used to be trained and then allow it to use either the cpu or the gpu.
model = MLP(input_size, hidden_size, output_size)
model.to(device)

criterion = nn.MSELoss()  # Suitable for classification tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train the model once everything has been initialized
print('Training model')
num_epochs = 10
for epoch in range(num_epochs):
    train_llm(training_dataloader, model, criterion, optimizer, epoch=epoch, num_epoch=num_epochs)
    validate_training(validation_dataloader, model)
