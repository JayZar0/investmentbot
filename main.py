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
training_data = StockTradingDataset('datasets/stock prices.csv', train=True)
validation_data = StockTradingDataset('datasets/stock prices.csv', train=False)

# create a predetermined batch size for the training dataset
batch_size = 4

# create the dataloader for training the llm
training_dataloader = DataLoader(training_data, batch_size=batch_size)
validation_dataloader = DataLoader(validation_data, batch_size=batch_size)

# set the layers size for the neural network out here rather than in the
# file/function itself
input_size = 784  # Example: for MNIST dataset (28x28 images)
hidden_size = 128
output_size = 10   # Example: 10 classes for classification

model = MLP(input_size, hidden_size, output_size)

# set the model to use the selected device so that if there is a gpu available
# it will use that for a much more faster and accurate training of the llm
criterion = nn.CrossEntropyLoss()  # Suitable for classification tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_llm(training_dataloader, model, criterion, optimizer)
validate_training(validation_dataloader, model)
