#!/usr/bin/env python3
"""
client_triton_fire.py – query a StockLSTM Triton deployment.

Uses the *same* Hydra YAML as the converter/engine-builder, so shapes and
server parameters stay in sync.
"""
from __future__ import annotations
import logging, sys
from pathlib import Path
from typing import List, Optional

import fire, numpy as np, tritonclient.http as httpclient
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from stocks_prediction.dataset.dataset_moex import MoexStockDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)


def _get_triton_client(url: str) -> httpclient.InferenceServerClient:
    try:
        cli = httpclient.InferenceServerClient(url=url, verbose=False)
        LOGGER.info("Connected to Triton @ %s", url)
        return cli
    except Exception:  # noqa: BLE001
        LOGGER.exception("Could not reach Triton")
        sys.exit(1)


def _run_batch(cli, model, arr):
    inp = httpclient.InferInput(name="input", shape=arr.shape, datatype="FP32")
    inp.set_data_from_numpy(arr)
    try:
        rsp = cli.infer(model_name=model, inputs=[inp])
        return rsp.as_numpy("output")
    except Exception:  # noqa: BLE001
        LOGGER.exception("Inference failed")
        raise


def query_triton(  # noqa: PLR0913
    *,
    data_dir: str,
    config: str = "conf/inference/stocklstm.yaml",
    server_url: Optional[str] = None,
    model_name: Optional[str] = None,
    seq_len: Optional[int] = None,
    horizon: Optional[int] = None,
    batch_size: Optional[int] = None,
    max_samples: Optional[int] = None,
    tickers: Optional[List[str]] = None,
) -> None:
    # --------------------------------------------------------------------- #
    cfg: DictConfig = OmegaConf.load(config)
    # CLI overrides → cfg
    for k, v in {
        "server_url": server_url,
        "model_name": model_name,
        "seq_len": seq_len,
        "horizon": horizon,
        "batch_size": batch_size,
        "max_samples": max_samples,
        "tickers": tickers,
    }.items():
        if v is not None:
            cfg[k] = v

    LOGGER.info(
        "cfg | server=%s  model=%s  seq_len=%d  batch=%d",
        cfg.server_url,
        cfg.model_name,
        cfg.seq_len,
        cfg.batch_size,
    )

    # --------------------------------------------------------------------- #
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

    loader = DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    cli = _get_triton_client(cfg.server_url)
    for X, y in loader:
        preds = _run_batch(cli, cfg.model_name, X.numpy().astype(np.float32))
        LOGGER.info("✓ pred=%.6f  truth=%.6f", float(preds[0]), float(y[0]))
        break


def main() -> None:  # pragma: no cover
    fire.Fire(query_triton)


if __name__ == "__main__":  # pragma: no cover
    main()
