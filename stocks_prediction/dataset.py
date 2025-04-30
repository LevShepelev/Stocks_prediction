import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


def load_data(
    file_path: Path,
    start_date: Union[str, datetime] = "1990-01-01",
    log_info: bool = False,
) -> np.ndarray:
    """
    Load a single Parquet file containing stock data and return the 'Close' column
    filtered by 'start_date'. The file must contain at least 'Date' and 'Close'.
    """

    if not file_path.exists() or file_path.stat().st_size == 0:
        logging.warning(f"{file_path} is empty or does not exist. Skipping.")
        raise ValueError(f"File {file_path} is empty or invalid.")

    # Convert start_date to datetime if it's a string
    if isinstance(start_date, str):
        try:
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError as exc:
            logging.error("start_date string is not in 'YYYY-MM-DD' format.")
            raise ValueError("Invalid start_date format.") from exc

    # Read the parquet file
    try:
        df = pl.read_parquet(file_path)
    except Exception as exc:
        logging.warning(f"Could not read file {file_path} as Parquet: {exc}")
        raise ValueError(f"Failed to read {file_path}.")

    # Ensure the file contains required columns
    if "Date" not in df.columns or "Close" not in df.columns:
        logging.warning(f"{file_path} does not contain 'Date' and 'Close' columns.")
        raise ValueError(f"Missing required columns in {file_path}.")

    if log_info:
        logging.info(f"Read {len(df)} rows from {file_path}.")

    # Ensure 'Date' is interpreted as a Polars datetime
    date_dtype = df["Date"].dtype

    if date_dtype == pl.Utf8:
        # If it's string, parse to pl.Date
        df = df.with_columns(
            pl.col("Date").str.strptime(pl.Date, "%Y-%m-%d").alias("Date")
        )
        # Optionally cast pl.Date to pl.Datetime if needed:
        df = df.with_columns(pl.col("Date").cast(pl.Datetime))
    elif date_dtype == pl.Date:
        # Already date; cast to datetime if you want a time-based type
        df = df.with_columns(pl.col("Date").cast(pl.Datetime))
    elif date_dtype == pl.Datetime:
        # It's already datetime, so do nothing
        pass
    else:
        # Possibly an unsupported type
        logging.warning(
            f"'Date' column in {file_path} is {date_dtype}, not string/date/datetime."
        )
        raise ValueError("Unsupported 'Date' column type for filtering.")

    # Filter rows by start_date
    df = df.filter(pl.col("Date") > start_date)

    if df.is_empty():
        logging.warning(f"{file_path} has no data after filtering by {start_date}.")
        raise ValueError(f"No data remains in {file_path} after filtering.")

    close_data = df["Close"].to_numpy()
    return close_data


def create_sliding_windows(
    data: np.ndarray, seq_length: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create (sequence, label) pairs from `data` using a sliding window approach.
    Returns a list of tuples (sequence, label).

    :param data: 1D NumPy array of float data.
    :param seq_length: The number of timesteps in each sequence window.
    :return: List of (sequence, label) tuples, where each sequence has shape (seq_length,)
             and each label is a single scalar.
    :raises ValueError: If `seq_length` is not positive or data is too short.
    """
    if seq_length < 1:
        raise ValueError("seq_length must be >= 1.")

    if len(data) < seq_length + 1:
        raise ValueError("Not enough data to create a single window.")

    # Use stride_tricks to create sliding windows efficiently
    windows = np.lib.stride_tricks.sliding_window_view(data, seq_length + 1)
    sequences = windows[:, :-1]
    labels = windows[:, -1]

    # Return a list of tuples (sequence, label)
    return list(zip(sequences, labels))


def load_all_data(
    directory: Path,
    start_date: Union[str, datetime] = "1990-01-01",
    seq_length: int = 60,
    log_info: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Load all .parquet files in the given directory, filter data by `start_date`,
    and create (sequence, label) pairs for each file. Each file must contain at least
    a 'Date' column (parseable as datetime) and a 'Close' column.

    :param directory: Path to the directory containing .parquet stock data files.
    :param start_date: The earliest date to keep. Can be a string 'YYYY-MM-DD' or datetime.
    :param seq_length: Number of timesteps per sequence window. Default is 60.
    :param log_info: If True, logs additional info-level messages.
    :return: A combined list of (sequence, label) tuples from all valid files.
    :raises ValueError: If no valid files or sequences are found.
    """

    all_pairs: List[Tuple[np.ndarray, np.ndarray]] = []

    parquet_files = list(directory.glob("*.parquet"))
    if not parquet_files:
        logging.warning(f"No .parquet files found in {directory}.")
        raise ValueError("No data files found.")

    # Use map/filter for concise iteration
    def process_file(fp: Path):
        try:
            data = load_data(fp, start_date, log_info)
            return create_sliding_windows(data, seq_length)
        except ValueError:
            # Already logged inside load_data, so just ignore
            return []

    # Gather all pairs from each file
    for pairs in map(process_file, parquet_files):
        all_pairs.extend(pairs)

    if not all_pairs:
        logging.warning("No valid sequences were created from any file.")
        raise ValueError("No sequences could be created.")
    print(len(all_pairs))
    return all_pairs


# dataset.py
class StockDataset(Dataset):
    def __init__(self, pairs: list[tuple[np.ndarray, np.ndarray]]):
        x, y = zip(*pairs)  # tuples → two lists
        self.x = torch.from_numpy(np.stack(x)).float().unsqueeze(-1)
        self.y = torch.from_numpy(np.stack(y)).float().unsqueeze(-1)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
