import logging
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.preprocessing import MinMaxScaler


def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(np.array(data).reshape(-1, 1))
    return data_scaled, scaler


def convert_txt_folder_to_parquet(
    txt_folder: str,
    parquet_folder: str,
    date_column: str = "Date",
    close_column: str = "Close",
) -> None:
    """
    Reads all .txt files from `txt_folder`, and for each file:
      1. Parses it as CSV.
      2. Ensures it has `date_column` and `close_column`.
      3. Writes out a corresponding .parquet file into `parquet_folder`.

    :param txt_folder: Path to a folder containing .txt files with columns including Date & Close.
    :param parquet_folder: Output folder where .parquet files will be saved.
    :param date_column: Name of the date column. Default is 'Date'.
    :param close_column: Name of the close column. Default is 'Close'.
    """

    txt_folder_path = Path(txt_folder)
    parquet_folder_path = Path(parquet_folder)
    parquet_folder_path.mkdir(parents=True, exist_ok=True)

    for txt_file in txt_folder_path.glob("*.txt"):
        try:
            # Read as CSV with polars
            df = pl.read_csv(
                str(txt_file),
                has_header=True,
            )
        except Exception as exc:
            logging.warning(f"Failed to read {txt_file} as CSV: {exc}")
            continue

        # Check columns
        if date_column not in df.columns or close_column not in df.columns:
            logging.warning(
                f"{txt_file} does not contain '{date_column}' and '{close_column}' columns. Skipping..."
            )
            continue

        # Convert date column to polars datetime if possible
        try:
            df = df.with_columns(
                pl.col(date_column).str.strptime(pl.Date, "%Y-%m-%d").alias(date_column)
            )

        except Exception as exc:
            logging.warning(
                f"Could not parse date column in {txt_file}. Skipping... {exc}"
            )
            continue

        # Write to parquet
        output_parquet_file = parquet_folder_path / (txt_file.stem + ".parquet")
        try:
            df.write_parquet(str(output_parquet_file))
            logging.info(f"Converted {txt_file} -> {output_parquet_file}")
        except Exception as exc:
            logging.warning(f"Failed to write Parquet for {txt_file}: {exc}")


def main():
    # Example usage:
    # Suppose your .txt files are in "./Data/Stocks"
    # and you want to convert them into "./Data/Stocks_parquet"
    txt_folder = "../Data/Stocks"
    parquet_folder = "../Data/Stocks_parquet"

    convert_txt_folder_to_parquet(txt_folder, parquet_folder)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
