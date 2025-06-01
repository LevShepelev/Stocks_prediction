#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Training entry-point for MOEX stock-prediction models.

Usage:
    $ poetry run python -m stocks_prediction.train \
    --config-path conf --config-name train_config
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Literal

import hydra
import mlflow
import pytorch_lightning as pl
import torch
from hydra.utils import get_original_cwd
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

# ─────────────────── local imports ────────────────────────────────────── #
from stocks_prediction.dataset.data_utils import build_dataloaders
from stocks_prediction.lightning_regressor import LightningRegressor
from stocks_prediction.mlflow_utils import fetch_and_plot_metrics, log_git_commit
from stocks_prediction.models import (
    LightningFEDformer,
    LightningInformer,
    LightningTimeSeriesTransformer,
    StockLSTM,
)


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="train_config")
def train(cfg: DictConfig) -> None:  # noqa: C901
    """Main training routine called by Hydra."""
    # 1 ─── Logging ─────────────────────────────────────────────────────── #
    level_name: str = cfg.get("Logging", {}).get("level", "INFO")
    logging.basicConfig(
        level=getattr(logging, level_name),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    logger.info("Merged Hydra config:\n%s", OmegaConf.to_yaml(cfg))

    # 2 ─── MLflow ──────────────────────────────────────────────────────── #
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    client = MlflowClient(tracking_uri=cfg.mlflow.tracking_uri)
    log_git_commit()

    # 3 ─── Data ────────────────────────────────────────────────────────── #
    dataset, train_loader, val_loader = build_dataloaders(cfg)

    # 4 ─── Model selection ─────────────────────────────────────────────── #
    model_type: Literal["stocklstm", "transformer", "fedformer", "informer"] = (
        cfg.model.type.lower()
    )
    if model_type == "stocklstm":
        core = StockLSTM(
            input_size=len(dataset.feature_cols),
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
        )
        lightning_module: pl.LightningModule = LightningRegressor(
            core, lr=cfg.training.learning_rate
        )

    elif model_type == "transformer":
        lightning_module = LightningTimeSeriesTransformer(
            enc_feat_dim=len(dataset.feature_cols),
            projection_dim=cfg.model.projection_dim,
            num_trans_blocks=cfg.model.num_trans_blocks,
            num_heads=cfg.model.num_heads,
            ff_dim=cfg.model.ff_dim,
            mlp_units=cfg.model.mlp_units,
            dropout=cfg.model.dropout,
            lr=cfg.training.learning_rate,
        )

    elif model_type == "fedformer":
        lightning_module = LightningFEDformer(
            OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True)),
            enc_feat_dim=len(dataset.feature_cols),
            learning_rate=cfg.training.learning_rate,
        )

    elif model_type == "informer":
        lightning_module = LightningInformer(
            OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True)),
            enc_feat_dim=len(dataset.feature_cols),
            learning_rate=cfg.training.learning_rate,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    # 5 ─── Trainer & callbacks ─────────────────────────────────────────── #
    mlf_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
        run_name=f"{cfg.model.type}_{Path(cfg.data.file).stem}",
    )
    ckpt_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="{epoch:02d}-{val_loss:.4f}",
        save_weights_only=True,
    )
    trainer = pl.Trainer(
        logger=mlf_logger,
        callbacks=[ckpt_cb, LearningRateMonitor(logging_interval="epoch")],
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
    )

    # 6 ─── Fit ─────────────────────────────────────────────────────────── #
    trainer.fit(lightning_module, train_loader, val_loader)

    # 7 ─── Persist artefacts ───────────────────────────────────────────── #
    best_ckpt = ckpt_cb.best_model_path
    mlf_logger.experiment.log_artifact(mlf_logger.run_id, best_ckpt)

    # Log *all* config values flattened for nice MLflow UI
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    def _flatten(d, parent_key=""):
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                yield from _flatten(v, new_key)
            else:
                yield new_key, v

    mlflow.log_params(dict(_flatten(cfg_dict)))

    # 8 ─── Metric plots ────────────────────────────────────────────────── #
    fetch_and_plot_metrics(
        client=client,
        run_id=mlf_logger.run_id,
        metric_keys=[
            "train_loss",
            "val_loss",
            "epoch",
            "lr-Adam",
        ],
        dst_dir=Path(get_original_cwd()) / "plots",
    )

    # 9 ─── Optional ONNX export (unchanged except for imports) ─────────── #
    try:
        onnx_path = Path(best_ckpt).with_suffix(".onnx")
        if model_type == "stocklstm":
            x, _ = next(iter(train_loader))
            lightning_module.to_onnx(
                file_path=onnx_path.as_posix(),
                input_sample=x[:1].cpu(),
                opset_version=19,
                export_params=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch", 1: "seq_len"},
                    "output": {0: "batch"},
                },
            )
        else:
            batch = next(iter(train_loader))
            label_len = getattr(cfg.model, "label_len", 0)
            pred_len = getattr(cfg.model, "pred_len", 1)
            x_enc, mark_enc, mark_dec, _ = batch
            x_dec = torch.cat(
                [x_enc[:, -label_len:, :], torch.zeros_like(x_enc[:, :pred_len, :])],
                dim=1,
            )
            dummy = (
                x_enc[:1].cpu(),
                mark_enc[:1].cpu(),
                x_dec[:1].cpu(),
                mark_dec[:1].cpu(),
            )
            lightning_module.to_onnx(
                file_path=onnx_path.as_posix(),
                input_sample=dummy,
                opset_version=19,
                export_params=True,
                simplify=True,
                input_names=["x_enc", "x_mark_enc", "x_dec", "x_mark_dec"],
                output_names=["output"],
                dynamic_axes={
                    "x_enc": {0: "batch", 1: "seq_len_enc"},
                    "x_mark_enc": {0: "batch", 1: "seq_len_enc"},
                    "x_dec": {0: "batch", 1: "seq_len_dec"},
                    "x_mark_dec": {0: "batch", 1: "seq_len_dec"},
                    "output": {0: "batch", 1: "seq_len_dec"},
                },
            )

        export_dir = Path(get_original_cwd()) / "exports" / "onnx"
        export_dir.mkdir(parents=True, exist_ok=True)
        final_name = f"{cfg.model.type}_{Path(cfg.data.file).stem}.onnx"
        shutil.copy2(onnx_path, export_dir / final_name)
        logger.info("ONNX model saved to %s", export_dir / final_name)
    except RuntimeError as exc:  # pragma: no cover
        logger.warning("ONNX export skipped: %s", exc)

    logger.info("Training completed ✔")


if __name__ == "__main__":
    train()
