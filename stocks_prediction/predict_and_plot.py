"""predict_and_plot_triton.py
--------------------------------
Query a StockLSTM *Triton* deployment and visualise the predictions against
MOEX ground‑truth prices.

Designed to be **import‑friendly** (callable from a future ``commands.py``)
while exposing a convenient CLI via **Fire**:

```bash
poetry run python predict_and_plot_triton.py \
    --data_dir ../data/raw \
    --server_url http://localhost:8011 \
    --model_name stocklstm \
    --seq_len 60 --horizon 1 \
    --batch_size 16 --max_samples 500
```

Key features
~~~~~~~~~~~~
* Re‑uses the low‑level client helpers from ``client_triton.py`` – no duplicated
  Triton logic.
* Automatically strips the URL **scheme** (``http://`` / ``https://``) so the
  Triton client always receives the correct *host:port* form.
* Strong typing, structured logging, docstrings, and graceful error handling.
* PEP 8 compliant, Black‑formatted, with explicit type hints and logging.
* Single public ``predict_and_plot`` function – ideal for reuse from other
  modules or CLI.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import fire
import matplotlib.pyplot as plt
import numpy as np
from stocks_prediction.dataset.dataset_moex import MoexStockDataset
from torch.utils.data import DataLoader, Subset

# Re‑use Triton helpers (local module)
from stocks_prediction.inference.triton.client_lstm import _get_triton_client, _run_batch  # type: ignore  # noqa: WPS450

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

__all__ = ["predict_and_plot"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_scheme(url: str) -> str:
    """Return *host:port* part of an URL, dropping the scheme if present."""
    parsed = urlparse(url)
    return parsed.netloc if parsed.scheme else url


def _load_dataset(
    *,
    data_dir: Path,
    seq_len: int,
    horizon: int,
    tickers: List[str],
    max_samples: Optional[int],
    batch_size: int,
) -> DataLoader:
    """Create a deterministic DataLoader for the requested MOEX subset."""
    dataset = MoexStockDataset(
        root_dir=data_dir,
        seq_len=seq_len,
        horizon=horizon,
        single_ticker=True,
        return_marks=False,
        tickers=tickers,
    )

    # Optionally cap the chronological sequence length
    if max_samples and max_samples < len(dataset):
        dataset = Subset(dataset, range(max_samples))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # chronological order matters
        num_workers=0,
        pin_memory=False,
    )


def _ensure_parent_dir(path: Path) -> None:
    """Create *parent* directories for *path* if they do not yet exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Public API (also exposed via Fire)
# ---------------------------------------------------------------------------

def predict_and_plot(  # noqa: PLR0913
    *,
    data_dir: str,
    server_url: str = "http://localhost:8011",
    model_name: str = "stocklstm",
    seq_len: int = 60,
    horizon: int = 1,
    batch_size: int = 32,
    max_samples: Optional[int] = None,
    tickers: Optional[List[str]] = None,
    outfile: str = "plots/predictions.png",
) -> None:
    """Query Triton for predictions and save a comparison plot.

    Parameters
    ----------
    data_dir : str
        Directory containing MOEX parquet files.
    server_url : str, default "http://localhost:8000"
        Triton inference server URL (scheme *optional*).
    model_name : str, default "stocklstm"
        Name of the deployed model on Triton.
    seq_len : int, default 60
        Input sequence length.
    horizon : int, default 1
        Forecast horizon (steps ahead).
    batch_size : int, default 64
        Batch size used when querying Triton.
    max_samples : int | None
        Cap on sequential samples (for speed). ``None`` → use all.
    tickers : list[str] | None
        Tickers to evaluate. Defaults to ``["GAZP"]``.
    outfile : str, default "plot/predictions.png"
        Output image path.
    """

    tickers = tickers or ["GAZP"]

    triton_host = _strip_scheme(server_url)

    LOGGER.info(
        "Using server=%s  model=%s  seq_len=%d  horizon=%d  batch=%d",
        triton_host,
        model_name,
        seq_len,
        horizon,
        batch_size,
    )

    # ------------------------------ data ---------------------------------- #
    loader = _load_dataset(
        data_dir=Path(data_dir),
        seq_len=seq_len,
        horizon=horizon,
        tickers=tickers,
        max_samples=max_samples,
        batch_size=batch_size,
    )

    # ----------------------------- client --------------------------------- #
    cli = _get_triton_client(triton_host)

    preds: List[float] = []
    truths: List[float] = []

    # --------------------------- inference -------------------------------- #
    for X, y in loader:
        X_np = X.numpy().astype(np.float32)
        try:
            batch_preds = _run_batch(cli, model_name, X_np)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Batch inference failed – aborting")
            raise exc from None

        if batch_preds.shape[0] != X_np.shape[0]:
            raise RuntimeError(
                f"Expected {X_np.shape[0]} predictions, got {batch_preds.shape[0]}",
            )

        preds.extend(batch_preds.squeeze().tolist())
        truths.extend(y.squeeze(-1).tolist())

    # ---------------------------- plotting -------------------------------- #
    _ensure_parent_dir(Path(outfile))
    LOGGER.info("Plotting %d predictions → %s", len(preds), outfile)

    plt.figure(figsize=(12, 6))
    plt.plot(truths, label="Ground Truth")
    plt.plot(preds, label="Predictions", alpha=0.7)

    title = (
        f"{model_name}: Predictions vs Ground Truth ({len(preds)} samples)"
        + (f" [limited to {max_samples}]" if max_samples else "")
    )
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Normalised Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)

    LOGGER.info("✓ wrote plot to %s", outfile)


# ---------------------------------------------------------------------------
# CLI entry‑point
# ---------------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    """Entry‑point for ``python predict_and_plot_triton.py``."""
    fire.Fire(predict_and_plot)


if __name__ == "__main__":  # pragma: no cover
    main()
