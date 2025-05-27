#!/usr/bin/env python3
"""
client_triton.py

Simple Triton HTTP client for the StockLSTM TensorRT plan.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tritonclient.http as httpclient
from torch.utils.data import DataLoader, Subset

from stocks_prediction.dataset_moex import MoexStockDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Triton client for StockLSTM TensorRT model"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to directory or file containing MOEX parquet data",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="localhost:8000",  # host:port only, no http://
        help="Triton server URL (host:port)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="stocklstm",
        help="Name of the Triton model",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=60,
        help="Sequence length used by the LSTM",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Prediction horizon (timesteps ahead)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size when querying the server",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on total samples to send",
    )
    return parser.parse_args()


def get_triton_client(server_url: str) -> httpclient.InferenceServerClient:
    """
    Create and return a Triton HTTP inference client.
    Expects server_url in the form "host:port" (no scheme).
    """
    try:
        client = httpclient.InferenceServerClient(server_url, verbose=False)
        logging.info("Connected to Triton at %s", server_url)
        return client
    except Exception:
        logging.exception("Failed to connect to Triton server")
        sys.exit(1)


def run_inference(
    client: httpclient.InferenceServerClient,
    model_name: str,
    input_array: np.ndarray,
) -> np.ndarray:
    """
    Send one batch to Triton and return the output predictions.
    """
    try:
        infer_input = httpclient.InferInput(
            name="input",
            shape=input_array.shape,
            datatype="FP32",
        )
        infer_input.set_data_from_numpy(input_array)
        response = client.infer(model_name=model_name, inputs=[infer_input])
        return response.as_numpy("output")
    except Exception:
        logging.exception("Inference request failed")
        raise


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
    )
    args = parse_args()

    # 1) Load dataset
    dataset = MoexStockDataset(
        root_dir=args.data_dir,
        seq_len=args.seq_len,
        horizon=args.horizon,
        single_ticker=True,
        return_marks=False,
        tickers=["GAZP"],
    )
    if args.max_samples and args.max_samples < len(dataset):
        dataset = Subset(dataset, list(range(args.max_samples)))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # 2) Connect to Triton (host:port, no http://)
    client = get_triton_client(args.server_url)

    # 3) Send one batch and print results
    for X_batch, y_batch in loader:
        np_input = X_batch.cpu().numpy().astype(np.float32)
        preds = run_inference(client, args.model_name, np_input)
        logging.info(
            "Received %d predictions; sample pred=%.6f, truth=%.6f",
            preds.shape[0],
            float(preds[0]),
            float(y_batch[0].item()),
        )
        break  # just test one batch

    logging.info("Client check succeeded — your Triton server is up and responsive.")


if __name__ == "__main__":
    main()
