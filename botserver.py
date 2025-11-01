from flask import Flask, jsonify, send_file
try:
    from flask_cors import CORS
    cors_available = True
except ImportError:
    cors_available = False
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
import io
import os
from datetime import datetime, timedelta
from neuralnetwork import MLP
from dataset import StockTradingDataset
from torch.utils.data import DataLoader

## this file will be used use the functions of the bot and tell the user which
## stocks to invest in. I may also make everything automated if I get too lazy
## to do the investments myself.

## I may also turn this to an api that can send graphs over to the front end.

app = Flask(__name__)

app = Flask(__name__)
if cors_available:
    CORS(app)  # Enable CORS for frontend integration

# Global variables for model and predictions
model = None
stock_predictions = {}
stock_data_history = []
error_corrections = []

# Initialize device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model():
    """Load the trained model if available"""
    global model
    if model is None:
        # Initialize model with same architecture as in main.py
        input_size = 2
        hidden_size = 256
        output_size = 5
        model = MLP(input_size, hidden_size, output_size)
        model.to(device)
        model.eval()
        # TODO: Load trained weights if available
        # model.load_state_dict(torch.load('model_weights.pth'))
    return model

# This function will be called on the start of the stock market opening to create
# the prediction of the stock market and which direction the stocks will take.
@app.route("/create")
def create_stock_data():
    """Create predictions for stock market at opening"""
    global stock_predictions
    model = load_model()
    
    try:
        # Load dataset to get stock codes and dates
        dataset = StockTradingDataset('datasets/stock prices modified.csv', train=False, device=device)
        
        # Get current date (market opening)
        today = datetime.now().date()
        
        # Get unique stock codes from dataset
        data = np.loadtxt('datasets/stock prices modified.csv', delimiter=',', dtype=str, skiprows=1)
        unique_stocks = np.unique(data[:, 0])
        
        predictions = {}
        for stock_code in unique_stocks[:10]:  # Limit to first 10 for demo
            # Find latest date for this stock in dataset
            stock_data = data[data[:, 0] == stock_code]
            if len(stock_data) > 0:
                latest_date = pd.to_datetime(stock_data[:, 1], format="%Y-%m-%d").max()
                
                # Create input for prediction (stock_code_index, timestamp)
                stock_codes, _ = np.unique(data[:, 0], return_inverse=True)
                stock_idx = np.where(stock_codes == stock_code)[0][0]
                date_timestamp = pd.to_datetime(str(today), format="%Y-%m-%d").astype(np.int64)
                
                # Make prediction
                input_tensor = torch.tensor([[float(stock_idx), float(date_timestamp)]], device=device)
                
                with torch.no_grad():
                    prediction = model(input_tensor).cpu().numpy()[0]
                
                predictions[stock_code] = {
                    'predicted_values': prediction.tolist(),
                    'date': str(today),
                    'direction': 'up' if prediction[0] > 0.5 else 'down'
                }
        
        stock_predictions = predictions
        return predictions
    except Exception as e:
        return {'error': str(e)}

# This function will be called after the stock data has been created. It will
# choose the stock based on the numbers that are shown in the stock data creation.
def select_stock():
    """Select the best stock to invest in based on predictions"""
    if not stock_predictions:
        create_stock_data()
    
    if not stock_predictions:
        return {'error': 'No stock data available'}
    
    # Select stock with highest predicted value (first output)
    best_stock = max(stock_predictions.items(), key=lambda x: x[1]['predicted_values'][0])
    
    return {
        'recommended_stock': best_stock[0],
        'prediction': best_stock[1],
        'all_predictions': stock_predictions
    }

# This function will be used to correct any errors that were made during the
# predictions that were made during the stock opening.
# It will only be called if the error margin was over 5%.
@app.route("/tune")
def correct_errors():
    """Correct prediction errors if margin > 5%"""
    global error_corrections
    corrections = []
    
    for stock_code, prediction_data in stock_predictions.items():
        predicted_value = prediction_data['predicted_values'][0]
        
        # Calculate error margin (assuming we have actual values to compare)
        # For now, simulate error calculation
        # TODO: Get actual stock values from API or database
        
        error_margin = abs(predicted_value - 0.5) * 100  # Simplified error calculation
        
        if error_margin > 5:
            # Apply correction
            correction_factor = 0.05  # 5% adjustment
            corrected_value = predicted_value * (1 - correction_factor if predicted_value > 0.5 else 1 + correction_factor)
            
            stock_predictions[stock_code]['predicted_values'][0] = corrected_value
            stock_predictions[stock_code]['corrected'] = True
            stock_predictions[stock_code]['original_value'] = predicted_value
            stock_predictions[stock_code]['correction_applied'] = correction_factor
            
            corrections.append({
                'stock': stock_code,
                'original_prediction': predicted_value,
                'corrected_prediction': corrected_value,
                'error_margin': error_margin
            })
    
    error_corrections.extend(corrections)
    return corrections

# This function will be used to save all the data of the stocks to create a
# data validation to see if the model is headed in the right direction.
@app.route("/close")
def end_of_trades():
    """Save stock data for validation at end of trading day"""
    global stock_data_history
    
    end_data = {
        'date': datetime.now().isoformat(),
        'predictions': stock_predictions.copy(),
        'corrections': error_corrections.copy()
    }
    
    stock_data_history.append(end_data)
    
    # Save to file for validation
    validation_file = 'validation_data.json'
    import json
    with open(validation_file, 'a') as f:
        f.write(json.dumps(end_data) + '\n')
    
    return {
        'status': 'Data saved successfully',
        'total_entries': len(stock_data_history)
    }

def generate_prediction_graph():
    """Generate prediction graph for visualization"""
    if not stock_predictions:
        create_stock_data()
    
    if not stock_predictions:
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    stocks = list(stock_predictions.keys())
    predictions = [stock_predictions[s]['predicted_values'][0] for s in stocks]
    
    # Create bar chart
    bars = ax.bar(stocks[:10], predictions[:10], color=['green' if p > 0.5 else 'red' for p in predictions[:10]])
    ax.set_xlabel('Stock Codes')
    ax.set_ylabel('Predicted Value')
    ax.set_title('Stock Market Predictions')
    ax.set_xticklabels(stocks[:10], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save to bytes buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close()
    
    return img_buffer

# Flask Routes

@app.route('/', methods=['GET'])
def index():
    """Home endpoint"""
    return jsonify({
        'message': 'Investment Bot API',
        'endpoints': {
            '/create_stock_data': 'Create stock market predictions',
            '/select_stock': 'Get recommended stock to invest in',
            '/correct_errors': 'Correct prediction errors',
            '/end_of_trades': 'Save trading data for validation',
            '/graph': 'Get prediction graph',
            '/status': 'Get current bot status'
        }
    })

@app.route('/create_stock_data', methods=['POST', 'GET'])
def create_stock_data_endpoint():
    """API endpoint to create stock data predictions"""
    predictions = create_stock_data()
    return jsonify({
        'status': 'success',
        'predictions': predictions,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/select_stock', methods=['GET'])
def select_stock_endpoint():
    """API endpoint to get recommended stock"""
    recommendation = select_stock()
    return jsonify(recommendation)

@app.route('/correct_errors', methods=['POST', 'GET'])
def correct_errors_endpoint():
    """API endpoint to correct prediction errors"""
    corrections = correct_errors()
    return jsonify({
        'status': 'success',
        'corrections': corrections,
        'total_corrections': len(corrections)
    })

@app.route('/end_of_trades', methods=['POST', 'GET'])
def end_of_trades_endpoint():
    """API endpoint to save trading data"""
    result = end_of_trades()
    return jsonify(result)

@app.route('/graph', methods=['GET'])
def graph_endpoint():
    """API endpoint to get prediction graph"""
    graph_buffer = generate_prediction_graph()
    if graph_buffer is None:
        return jsonify({'error': 'No predictions available'}), 400
    return send_file(graph_buffer, mimetype='image/png')

@app.route('/status', methods=['GET'])
def status_endpoint():
    """Get current bot status"""
    return jsonify({
        'model_loaded': model is not None,
        'device': device,
        'predictions_count': len(stock_predictions),
        'history_entries': len(stock_data_history),
        'corrections_count': len(error_corrections)
    })

if __name__ == '__main__':
    print(f'Starting Investment Bot Server on {device}')
    app.run(debug=True, host='0.0.0.0', port=5000)