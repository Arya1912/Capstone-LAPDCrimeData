import pandas as pd
import numpy as np

def create_time_series_sequences(df, lookback=14):
    """
    Groups incidents by day and cluster to create sliding windows.
    Each window contains 14 days of history (X) and 1 target day (y).
    """
    # 1. Aggregate incidents per day per cluster
    df['DATE OCC'] = pd.to_datetime(df['DATE OCC'])
    daily_counts = df.groupby(['DATE OCC', 'cluster_id']).size().unstack(fill_value=0)
    
    X, y = [], []
    data_values = daily_counts.values
    
    # 2. Create the Sliding Window (the 'Lookback')
    for i in range(len(data_values) - lookback):
        X.append(data_values[i : i + lookback]) # Memory
        y.append(data_values[i + lookback])     # Target
        
    return np.array(X), np.array(y), daily_counts.columns

def scale_sequences(X, y):
    """Normalizes sequences between 0 and 1 for optimal deep learning performance."""
    # Min-Max Scaling ensures the LSTM converges faster (9-minute efficiency)
    x_min, x_max = X.min(), X.max()
    X_scaled = (X - x_min) / (x_max - x_min)
    y_scaled = (y - x_min) / (x_max - x_min)
    
    return X_scaled, y_scaled, x_min, x_max