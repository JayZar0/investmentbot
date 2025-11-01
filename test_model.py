import torch
import numpy as np
import pandas as pd
import pickle
from neuralnetwork import MLP
from dataset import StockTradingDataset

## This file is used to test the trained model on new data

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load the saved model
print('Loading trained model')
model_save_path = 'models/trained_model.pth'
checkpoint = torch.load(model_save_path, map_location=device)

# Recreate the model architecture
model = MLP(
    input_size=checkpoint['input_size'],
    hidden_size=checkpoint['hidden_size'],
    output_size=checkpoint['output_size']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
print('Model loaded successfully')

# Load scalers for preprocessing
scaler_dir = 'scalers'
with open(f'{scaler_dir}/input_scaler.pkl', 'rb') as f:
    input_scaler = pickle.load(f)
with open(f'{scaler_dir}/output_scaler.pkl', 'rb') as f:
    output_scaler = pickle.load(f)
with open(f'{scaler_dir}/date_scaler.pkl', 'rb') as f:
    date_scaler_info = pickle.load(f)
with open(f'{scaler_dir}/stock_encoder.pkl', 'rb') as f:
    stock_encoder = pickle.load(f)
    all_unique_stocks = pickle.load(f)
    stock_to_idx = pickle.load(f)

print('Scalers loaded successfully')

# Load validation data for testing
print('Loading validation data')
validation_data = StockTradingDataset('datasets/stock prices modified.csv', train=False, device=device)
print(f'Validation samples: {len(validation_data)}')

# Test on a batch of validation data
print('\nTesting on validation data...')
num_test_samples = min(10, len(validation_data))
test_indices = np.random.choice(len(validation_data), num_test_samples, replace=False)

predictions = []
actuals = []
stocks = []
dates = []

with torch.no_grad():
    for idx in test_indices:
        inputs, labels = validation_data[idx]
        outputs = model(inputs.unsqueeze(0))
        
        # Inverse transform to get actual values
        pred = output_scaler.inverse_transform(outputs.cpu().numpy())
        actual = output_scaler.inverse_transform(labels.cpu().numpy().reshape(1, -1))
        
        predictions.append(pred[0])
        actuals.append(actual[0])
        
        # Get stock and date information
        stocks.append(validation_data.stock_codes[idx])
        dates.append(validation_data.stock_codes[idx])  # This would need date info from dataset
        
        # Calculate and print individual prediction accuracy
        mae = np.mean(np.abs(pred - actual))
        print(f'\nStock: {stocks[-1]}, Date: {dates[-1]}')
        print(f'  Predicted: Open={pred[0][0]:.2f}, High={pred[0][1]:.2f}, Low={pred[0][2]:.2f}, Close={pred[0][3]:.2f}, Volume={pred[0][4]:.2f}')
        print(f'  Actual:    Open={actual[0][0]:.2f}, High={actual[0][1]:.2f}, Low={actual[0][2]:.2f}, Close={actual[0][3]:.2f}, Volume={actual[0][4]:.2f}')
        print(f'  MAE: {mae:.4f}')

# Calculate overall statistics
predictions = np.array(predictions)
actuals = np.array(actuals)

mae_overall = np.mean(np.abs(predictions - actuals))
mse_overall = np.mean((predictions - actuals) ** 2)
rmse_overall = np.sqrt(mse_overall)

print('\n' + '='*60)
print('OVERALL TEST STATISTICS')
print('='*60)
print(f'Number of test samples: {num_test_samples}')
print(f'Mean Absolute Error (MAE): {mae_overall:.4f}')
print(f'Mean Squared Error (MSE): {mse_overall:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse_overall:.4f}')

# Calculate per-feature statistics
feature_names = ['Open', 'High', 'Low', 'Close', 'Volume']
print('\nPer-feature MAE:')
for i, name in enumerate(feature_names):
    feature_mae = np.mean(np.abs(predictions[:, i] - actuals[:, i]))
    print(f'  {name}: {feature_mae:.4f}')

print('\nTest completed!')

