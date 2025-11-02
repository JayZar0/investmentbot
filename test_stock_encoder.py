"""
Quick test to verify stock code encoding works correctly.
This verifies that stock codes are properly recognized as the primary prediction source.
"""

import numpy as np
import pandas as pd
from dataset import StockTradingDataset

# Try to load the dataset to see what stocks are available
print("Loading dataset to check stock code encoding...")
try:
    # Load a small sample just to initialize
    training_data = StockTradingDataset('datasets/stock prices modified.csv', train=True, device='cpu')
    
    # Get stock codes
    stock_codes = training_data.get_stock_codes()
    print(f"\nFound {len(stock_codes)} unique stocks in dataset")
    print(f"First 10 stocks: {stock_codes[:10]}")
    
    # Test the static method for preparing predictions
    if len(stock_codes) > 0:
        test_stock = stock_codes[0]
        test_date = "2024-01-15"
        
        print(f"\nTesting prediction input preparation for stock: {test_stock}")
        try:
            features = StockTradingDataset.prepare_stock_code_for_prediction(
                test_stock, test_date, scaler_dir='scalers'
            )
            print(f"✓ Successfully prepared features")
            print(f"  Feature shape: {features.shape}")
            print(f"  First 10 values: {features[:10]}")
            print(f"  Non-zero entries (should be 1 for one-hot stock code): {np.count_nonzero(features == 1.0)}")
            
            # Also test instance method
            features2 = training_data.prepare_prediction_input(test_stock, test_date)
            if np.allclose(features, features2):
                print("✓ Instance method matches static method")
            else:
                print("✗ Instance method differs from static method")
                
        except Exception as e:
            print(f"✗ Error preparing features: {e}")
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete!")

