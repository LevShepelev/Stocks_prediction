#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Universal training entry-point for LSTM, Transformer and FEDformer models.

Run e.g.:
    python train_loop.py --config-path conf --config-name train_config
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Tuple
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import hydra
import mlflow
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

# local imports
from stocks_prediction.dataset_moex import MoexStockDataset
from stocks_prediction.models import (
    StockLSTM,
    TimeSeriesTransformerWithProjection,
    LightningFEDformer,  # wrapper we created earlier
)

# -----------------------------------------------------------------------------#
#                       Generic regression wrapper (LSTM / Transformer)        #
# -----------------------------------------------------------------------------#
class LightningRegressor(pl.LightningModule):
    """Wrap any nn.Module that returns (batch, 1) for regression."""

    def __init__(self, model: nn.Module, lr: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:  # noqa: D401,E501
        x, y = batch
        y_hat = self(x).squeeze(-1)
        loss = self.criterion(y_hat, y.squeeze(-1))
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _):
        x, y = batch
        y_hat = self(x).squeeze(-1)
        loss = self.criterion(y_hat, y.squeeze(-1))
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)  # type: ignore


# -----------------------------------------------------------------------------#
#                            Training entry-point                              #
# -----------------------------------------------------------------------------#
@hydra.main(version_base=None, config_path="conf", config_name="train_config")
def train(cfg: DictConfig) -> None:
    # 1 ─── Logging -----------------------------------------------------------
    level_name: str = cfg.get("Logging", {}).get("level", "INFO")
    logging.basicConfig(
        level=getattr(logging, level_name),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    logging.info("Merged Hydra config:\n%s", OmegaConf.to_yaml(cfg))

    # 2 ─── MLflow ------------------------------------------------------------
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # 3 ─── Dataset -----------------------------------------------------------
    model_type = cfg.model.type.lower()
    need_marks = model_type == "fedformer"

    ds_kwargs: dict[str, Any] = dict(
        root_dir=Path(cfg.data.file),
        seq_len=cfg.model.seq_len if need_marks else cfg.data.seq_length,
        horizon=cfg.data.horizon,
        single_ticker=True,
        stride=cfg.data.stride,
        return_marks=need_marks,
    )
    if need_marks:
        ds_kwargs.update(
            label_len=cfg.model.label_len,
            pred_len=cfg.model.pred_len,
        )

    full_dataset = MoexStockDataset(**ds_kwargs)
    train_size = int(len(full_dataset) * cfg.training.train_val_split)
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    # 4 ─── Model -------------------------------------------------------------
    if model_type == "lstm":
        core = StockLSTM(
            input_size=len(full_dataset.feature_cols),
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
        )
        lightning_module: pl.LightningModule = LightningRegressor(
            core, lr=cfg.training.learning_rate
        )

    elif model_type == "transformer":
        core = TimeSeriesTransformerWithProjection(
            projection_dim=cfg.model.projection_dim,
            num_trans_blocks=cfg.model.num_trans_blocks,
            num_heads=cfg.model.num_heads,
            ff_dim=cfg.model.ff_dim,
            mlp_units=cfg.model.mlp_units,
            dropout=cfg.model.dropout,
        )
        lightning_module = LightningRegressor(core, lr=cfg.training.learning_rate)

    elif model_type == "fedformer":
        fed_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        lightning_module = LightningFEDformer(
            fed_cfg,
            enc_feat_dim=len(full_dataset.feature_cols),
            learning_rate=cfg.training.learning_rate,
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    # 5 ─── Trainer -----------------------------------------------------------
    mlf_logger = MLFlowLogger(
    experiment_name=cfg.mlflow.experiment_name,
    tracking_uri=cfg.mlflow.tracking_uri,
    run_name=f"{cfg.model.type}_{Path(cfg.data.file).stem}",
    )   
    mlf_logger = MLFlowLogger(
    experiment_name=cfg.mlflow.experiment_name,
    tracking_uri=cfg.mlflow.tracking_uri,
    run_name=f"{cfg.model.type}_{Path(cfg.data.file).stem}",
    )
    ckpt_cb = ModelCheckpoint(
    monitor="val_loss",          # tracked metric
    mode="min",                  # smaller is better
    save_top_k=1,
    filename="{epoch:02d}-{val_loss:.4f}",
    )
    trainer = pl.Trainer(
        logger=mlf_logger,
        callbacks=[ckpt_cb],         # ← add callback
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        gradient_clip_val=0.5, gradient_clip_algorithm="norm"
    )

    # 6 ─── MLflow run --------------------------------------------------------

    trainer.fit(lightning_module, train_loader, val_loader)
    # after training ends
    best_ckpt = ckpt_cb.best_model_path
    mlf_logger.experiment.log_artifact(mlf_logger.run_id, best_ckpt)

    # save the exact Hydra config used
    cfg_file = Path(hydra.utils.get_original_cwd()) / "train_run_cfg.yaml"
    OmegaConf.save(cfg, cfg_file)
    mlf_logger.experiment.log_artifact(mlf_logger.run_id, cfg_file)

    # optional: ONNX, TensorRT, plots, csv, …

    logging.info("Training completed ✔")


if __name__ == "__main__":
    train()
