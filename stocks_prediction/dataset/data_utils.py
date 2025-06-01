"""Dataset creation & time-series-aware split helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Tuple

from omegaconf import DictConfig
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, Subset

from stocks_prediction.dataset.dataset_moex import MoexStockDataset


logger = logging.getLogger(__name__)


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


def time_series_split(
    dataset: MoexStockDataset, *, n_splits: int
) -> Tuple[Subset, Subset]:
    """
    Perform **chronological** Train/Val split using scikit-learnʼs
    :class:`~sklearn.model_selection.TimeSeriesSplit`.

    Returns
    -------
    (train_subset, val_subset)
    """
    if n_splits < 2:
        raise ValueError("``n_splits`` must be ≥ 2 for TimeSeriesSplit.")

    tss = TimeSeriesSplit(n_splits=n_splits, gap=0)
    # Take the **last** split to mimic a single hold-out validation set
    train_idx, val_idx = list(tss.split(range(len(dataset))))[-1]

    logger.info(
        "TimeSeriesSplit: train=%d samples, val=%d samples",
        len(train_idx),
        len(val_idx),
    )
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def build_dataloaders(
    cfg: DictConfig,
) -> Tuple[MoexStockDataset, DataLoader, DataLoader]:
    """Return dataset + chronologically-correct loaders."""
    ds = build_dataset(cfg)
    train_ds, val_ds = time_series_split(ds, n_splits=cfg.training.n_splits)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,  # DO NOT shuffle time-series!
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
