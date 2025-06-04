"""stocks_prediction.dataset.split_cached
================================================
Chronological train/validation split helpers **with on‑disk caching**.

This module guarantees that a given dataset is always split in exactly the
same way (deterministic `TimeSeriesSplit` **last fold**) *and* that the chosen
indices are re‑used across machines/CI runs.

Usage
-----
>>> from pathlib import Path
>>> from stocks_prediction.dataset.split_cached import (
...     build_dataset,
...     build_dataloaders,
... )
>>> ds, tr_loader, val_loader = build_dataloaders(cfg)

The helpers respect two extra keys in your Hydra/OmegaConf config under
`training`:

```
training:
  n_splits: 5            # already existing
  split_cache_dir: "splits"  # optional; defaults to "splits"
```

All artefacts are stored as

```
<split_cache_dir>/
    moex-train.npy
    moex-val.npy
```

They are **tiny** (<10 kB) and safe to commit.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Final

import numpy as np
from omegaconf import DictConfig
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, Subset

from stocks_prediction.dataset.dataset_moex import MoexStockDataset


logger = logging.getLogger(__name__)

#: File‑stem used when writing the *.npy* index artefacts
_STEM: Final[str] = "moex"


# ---------------------------------------------------------------------------
# Split‑caching helpers
# ---------------------------------------------------------------------------


def _last_fold_indices(
    n_samples: int, *, n_splits: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return train / val indices of the **last** fold (chronological hold‑out)."""
    if n_splits < 2:
        raise ValueError("`n_splits` must be ≥ 2.")
    tss = TimeSeriesSplit(n_splits=n_splits, gap=0)
    train_idx, val_idx = list(tss.split(range(n_samples)))[-1]
    return np.asarray(train_idx, dtype=np.int32), np.asarray(val_idx, dtype=np.int32)


def _index_files(cache_dir: Path) -> tuple[Path, Path]:
    """Helper that returns the *(train_file, val_file)* paths."""
    return cache_dir / f"{_STEM}-train.npy", cache_dir / f"{_STEM}-val.npy"


def _get_or_create_indices(
    n_samples: int,
    *,
    n_splits: int,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load cached indices or generate & persist them."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_f, val_f = _index_files(cache_dir)

    if train_f.exists() and val_f.exists():
        train_idx, val_idx = np.load(train_f), np.load(val_f)
        logger.info("Loaded cached split from %s", cache_dir)

        # If the dataset grew/shrank, regenerate to avoid shape mismatch.
        if len(train_idx) + len(val_idx) != n_samples:
            logger.warning(
                "Dataset length changed → regenerating cached split (%d → %d).",
                len(train_idx) + len(val_idx),
                n_samples,
            )
            train_idx, val_idx = _last_fold_indices(n_samples, n_splits=n_splits)
            np.save(train_f, train_idx)
            np.save(val_f, val_idx)
    else:
        train_idx, val_idx = _last_fold_indices(n_samples, n_splits=n_splits)
        np.save(train_f, train_idx)
        np.save(val_f, val_idx)
        logger.info("Saved new split to %s", cache_dir)

    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Public API re‑exported by module
# ---------------------------------------------------------------------------


def build_dataset(cfg: DictConfig) -> MoexStockDataset:
    """Instantiate ``MoexStockDataset`` with flags inferred from ``cfg``."""
    model_type: str = cfg.model.type.lower()
    need_marks = model_type in {"fedformer", "informer", "transformer"}

    kwargs: dict[str, Any] = {
        "root_dir": Path(cfg.data.file),
        "seq_len": cfg.data.seq_length,
        "horizon": cfg.data.horizon,
        "single_ticker": True,
        "stride": cfg.data.stride,
        "return_marks": need_marks,
    }
    if need_marks:
        kwargs.update(label_len=cfg.model.label_len, pred_len=cfg.model.pred_len)

    return MoexStockDataset(**kwargs)


def time_series_split_cached(
    dataset: MoexStockDataset,
    *,
    n_splits: int,
    cache_dir: Path = Path("splits"),
) -> tuple[Subset, Subset]:
    """Chronological split backed by on‑disk cache (see module docs)."""
    train_idx, val_idx = _get_or_create_indices(
        len(dataset), n_splits=n_splits, cache_dir=cache_dir
    )
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def build_dataloaders(
    cfg: DictConfig,
) -> tuple[MoexStockDataset, DataLoader, DataLoader]:
    """Return dataset plus *deterministic* train/val ``DataLoader``s."""

    ds = build_dataset(cfg)

    cache_dir = (
        Path(cfg.training.split_cache_dir)
        if getattr(cfg.training, "split_cache_dir", None)
        else Path("splits")
    )

    train_ds, val_ds = time_series_split_cached(
        ds, n_splits=cfg.training.n_splits, cache_dir=cache_dir
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,  # DO NOT shuffle chronological data!
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    return ds, train_loader, val_loader


# ---------------------------------------------------------------------------
# Module tests (pytest‑discoverable) – run with `pytest -q`
# ---------------------------------------------------------------------------


def _fake_config(tmp_path: Path) -> DictConfig:  # pragma: no cover – helper
    """Minimal DictConfig stub for quick unit tests."""
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "data": {
                "file": str(tmp_path / "stub.csv"),
                "seq_length": 16,
                "horizon": 8,
                "stride": 1,
            },
            "model": {
                "type": "informer",
                "label_len": 16,
                "pred_len": 8,
            },
            "training": {
                "n_splits": 4,
                "batch_size": 32,
                "num_workers": 0,
                "split_cache_dir": str(tmp_path / "splits"),
            },
        }
    )


def test_get_or_create_indices(tmp_path: Path) -> None:  # pragma: no cover
    """Indices must be stable across two consecutive calls."""
    n_samples, n_splits = 100, 4
    cache_dir = tmp_path / "splits"

    idx1 = _get_or_create_indices(n_samples, n_splits=n_splits, cache_dir=cache_dir)
    idx2 = _get_or_create_indices(n_samples, n_splits=n_splits, cache_dir=cache_dir)

    assert (idx1[0] == idx2[0]).all() and (idx1[1] == idx2[1]).all()
