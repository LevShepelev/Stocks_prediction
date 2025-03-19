import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler


def load_data(file_path, start_date="1990-01-01"):
    """Load a single CSV/TXT file and return the 'Close'
    column filtered by start_date."""
    # Check if file is empty
    if os.stat(file_path).st_size == 0:
        print(f"Warning: {file_path} is empty. Skipping.")
        return None
    try:
        df = pd.read_csv(file_path, sep=",")
    except pd.errors.EmptyDataError:
        print(f"Warning: {file_path} contains no data. Skipping.")
        return None
    if df.empty:
        print(f"Warning: {file_path} read as empty DataFrame. Skipping.")
        return None
    # Ensure the file contains required columns
    if "Date" not in df.columns or "Close" not in df.columns:
        print(f"Warning: {file_path} does not contain required columns. Skipping.")
        return None
    df = df[df["Date"] > start_date]
    return df["Close"]


def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length - 1):
        sequences.append(data[i : (i + seq_length)])
        labels.append(data[i + seq_length])
    # Convert lists to NumPy arrays and then to tensors
    sequences_np = np.array(sequences)
    labels_np = np.array(labels)
    return torch.tensor(sequences_np, dtype=torch.float32), torch.tensor(
        labels_np, dtype=torch.float32
    )


def preprocess_data(data):
    """Scale the data to the range [0,1]."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(np.array(data).reshape(-1, 1))
    return data_scaled, scaler


def load_all_data(directory, start_date="1990-01-01", seq_length=60):
    """
    Load all .txt files in the given directory (assuming they contain stock data),
    process each one, and concatenate the sequences from all files.
    """
    all_sequences = []
    all_labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            data = load_data(file_path, start_date)
            if data is None:
                continue  # Skip files that couldn’t be loaded properly
            if len(data) <= seq_length:
                continue  # Skip files with insufficient data
            data_scaled, scaler = preprocess_data(data)
            X, y = create_sequences(data_scaled, seq_length)
            all_sequences.append(X)
            all_labels.append(y)
    if all_sequences:
        X_all = torch.cat(all_sequences, dim=0)
        y_all = torch.cat(all_labels, dim=0)
        return X_all, y_all
    else:
        return None, None
