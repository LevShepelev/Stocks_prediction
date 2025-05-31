#!/usr/bin/env python3
"""
onnx_client_fire.py
-------------------
Call the FastAPI endpoint of `onnx_server_fire.py` and print a quick sanity-check.

Example
-------
poetry run python inference/onnx_client_fire.py \
    --data_dir data/moex \
    batch_size=8 port=8020       # CLI overrides
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import fire
import numpy as np
import requests
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from stocks_prediction.dataset.dataset_moex import MoexStockDataset


###############################################################################
# Logging
###############################################################################
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)


###############################################################################
# Client logic
###############################################################################
def _post_batch(url: str, arr: np.ndarray) -> List[float]:
    """POST ‹arr› to *url* and return the prediction list."""
    payload = {"input_sequence": arr.tolist()}
    try:
        rsp = requests.post(url, data=json.dumps(payload), timeout=30)
        rsp.raise_for_status()
        return rsp.json()["prediction"]
    except requests.exceptions.RequestException as exc:  # noqa: BLE001
        LOGGER.error("HTTP error: %s", exc)
        raise


###############################################################################
# Fire entry-point
###############################################################################
def query_server(  # noqa: PLR0913
    *,
    data_dir: str,
    config: str = "conf/inference/onnx_stocklstm.yaml",
    host: str | None = None,
    port: int | None = None,
    seq_len: int | None = None,
    horizon: int | None = None,
    batch_size: int | None = None,
    max_samples: int | None = None,
    tickers: Optional[List[str]] = None,
) -> None:
    """
    Send one batch of data to the ONNX server and print a sample prediction.

    Any CLI flag overrides the same-named key in *config*.
    """
    # ------------------------------------------------------------------ #
    # 1  Load + merge config
    # ------------------------------------------------------------------ #
    cfg: DictConfig = OmegaConf.load(config)
    for k, v in {
        "host": host,
        "port": port,
        "seq_len": seq_len,
        "horizon": horizon,
        "batch_size": batch_size,
        "max_samples": max_samples,
        "tickers": tickers,
    }.items():
        if v is not None:
            cfg[k] = v

    base_url = f"http://{cfg.host}:{cfg.port}/predict"
    LOGGER.info("⇢  POSTing to %s", base_url)

    # ------------------------------------------------------------------ #
    # 2  Dataset + loader
    # ------------------------------------------------------------------ #
    ds = MoexStockDataset(
        root_dir=Path(data_dir),
        seq_len=int(cfg.seq_len),
        horizon=int(cfg.horizon),
        single_ticker=True,
        return_marks=False,
        tickers=cfg.tickers,
    )
    if cfg.max_samples and cfg.max_samples < len(ds):
        ds = Subset(ds, range(cfg.max_samples))

    dl = DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # ------------------------------------------------------------------ #
    # 3  Send ONE batch as a smoke test
    # ------------------------------------------------------------------ #
    for X, y in dl:
        preds = _post_batch(base_url, X.numpy().astype(np.float32))
        LOGGER.info(
            "✓ got %d preds – sample: pred=%.6f  truth=%.6f",
            len(preds),
            float(preds[0]),
            float(y[0]),
        )
        break  # stop after first batch

    LOGGER.info("ONNX server client check succeeded.")


def main() -> None:  # pragma: no cover
    fire.Fire(query_server)


if __name__ == "__main__":  # pragma: no cover
    main()
