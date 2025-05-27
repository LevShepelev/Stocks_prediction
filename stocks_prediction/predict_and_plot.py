# predict_and_plot.py
# Script to load MOEX data, query the StockLSTM ONNX server for next-step predictions,
# and plot predictions vs. ground truth, with optional limit on total samples.

import os
import argparse
import requests
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, Subset

from stocks_prediction.dataset_moex import MoexStockDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load MOEX data, call ONNX server for predictions, and plot results"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Path to directory or file containing MOEX parquet data"
    )
    parser.add_argument(
        "--seq-len", type=int, default=60,
        help="Sequence length used by the LSTM"
    )
    parser.add_argument(
        "--horizon", type=int, default=1,
        help="Prediction horizon"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size when querying the server"
    )
    parser.add_argument(
        "--server-url", type=str,
        default=os.environ.get("SERVER_URL", "http://localhost:8020/predict"),
        help="Full URL of the ONNX server's /predict endpoint"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Maximum number of samples to predict (default: all available samples)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Load dataset
    dataset = MoexStockDataset(
        root_dir=Path(args.data_dir),
        seq_len=args.seq_len,
        horizon=args.horizon,
        single_ticker=True,
        return_marks=False,
        tickers=["GAZP"]
    )

    # 1a) Optionally limit to first N samples via Subset
    if args.max_samples is not None and args.max_samples < len(dataset):
        dataset = Subset(dataset, list(range(args.max_samples)))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # preserve chronological order
        num_workers=0,   # set to 0 to avoid pin_memory hang
        pin_memory=False,
    )

    all_preds = []
    all_gt = []

    # 2) Iterate and call server
    for X, y in loader:
        # X shape: (batch, seq_len, n_features)
        seqs = X.cpu().tolist()
        payload = {"input_sequence": seqs}
        response = requests.post(args.server_url, json=payload)
        response.raise_for_status()
        preds = response.json().get("prediction", [])
        if len(preds) != X.size(0):
            raise RuntimeError(f"Expected {X.size(0)} predictions, got {len(preds)}")

        all_preds.extend(preds)
        all_gt.extend(y.squeeze(-1).cpu().tolist())

    # 3) Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(all_gt, label="Ground Truth")
    plt.plot(all_preds, label="Predictions", alpha=0.7)
    title = f"StockLSTM: Predictions vs Ground Truth ({len(all_preds)} samples)"
    if args.max_samples:
        title += f" [limited to {args.max_samples}]"
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot.png")


if __name__ == "__main__":
    main()