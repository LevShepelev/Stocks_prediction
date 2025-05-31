#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_loop.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Universal training entry‑point for MOEX stock‑prediction models (LSTM,
Transformer family, FEDformer, Informer).

**What is new in this version?**
    • Structured logging to *stdout* **and** MLflow.
    • Automatic MLflow–ML plots export to a local *plots/* directory (no need
      to create graphs manually!).
    • Hyper‑parameters, git commit id and code version tracked in every run.
    • Better error‑handling, callbacks (checkpoints, LR monitor), PEP‑8,
      typing, and Black‑compatible formatting.

Usage
-----
    >>> python train_loop.py --config-path conf --config-name train_config
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Tuple

import hydra
import matplotlib.pyplot as plt
import mlflow
import pytorch_lightning as pl
import torch
from hydra.utils import get_original_cwd
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

# ─── Local imports ────────────────────────────────────────────────────────── #
from stocks_prediction.dataset.dataset_moex import MoexStockDataset
from stocks_prediction.models import (
    LightningFEDformer,
    LightningInformer,
    LightningTimeSeriesTransformer,
    StockLSTM,
)


# --------------------------------------------------------------------------- #
#                         Generic regression wrapper
# --------------------------------------------------------------------------- #
#                         Generic regression wrapper
# --------------------------------------------------------------------------- #


class LightningRegressor(pl.LightningModule):
    """Wrap any ``nn.Module`` that returns a 1‑D regression output."""

    def __init__(self, model: nn.Module, lr: float = 1e-3) -> None:  # noqa: D401
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.criterion = nn.MSELoss()

    # --------------------------------------------------------------------- #
    #                            Forward / Steps                           #
    # --------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.model(x)

    def _shared_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self(x).squeeze(-1)
        loss = self.criterion(y_hat, y.squeeze(-1))
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], _
    ) -> torch.Tensor:  # noqa: E501
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _):
        self._shared_step(batch, "val")

    # --------------------------------------------------------------------- #
    #                        Optimiser / Scheduler                          #
    # --------------------------------------------------------------------- #

    def configure_optimizers(self):  # noqa: D401
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }


# --------------------------------------------------------------------------- #
#                       Auxiliary: plots export helper                       #
# --------------------------------------------------------------------------- #


def _fetch_and_plot_metrics(
    client: MlflowClient,
    run_id: str,
    metric_keys: Iterable[str],
    dst_dir: Path,
) -> None:
    """Download metric history from MLflow and store PNG plots in *dst_dir*."""

    dst_dir.mkdir(parents=True, exist_ok=True)

    for key in metric_keys:
        history = client.get_metric_history(run_id, key)
        if not history:
            logging.warning("No metric history for key '%s'", key)
            continue
        steps = [p.step for p in history]
        values = [p.value for p in history]

        plt.figure()
        plt.plot(steps, values)
        plt.xlabel("step")
        plt.ylabel(key)
        plt.title(key)
        plt.tight_layout()

        fig_path = dst_dir / f"{key}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()

        # add to MLflow artefacts for convenience --------------------------------
        mlflow.log_artifact(fig_path, artifact_path="plots")


# --------------------------------------------------------------------------- #
#                              Training entry‑point                           #
# --------------------------------------------------------------------------- #


@hydra.main(version_base=None, config_path="conf", config_name="train_config")
def train(cfg: DictConfig) -> None:  # noqa: C901 (core function)
    """Main training routine called by Hydra."""

    # 1 ── Logging ---------------------------------------------------------- #
    level_name: str = cfg.get("Logging", {}).get("level", "INFO")
    logging.basicConfig(
        level=getattr(logging, level_name),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    logging.info("Merged Hydra config:\n%s", OmegaConf.to_yaml(cfg))

    # 2 ── MLflow ----------------------------------------------------------- #
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # Create an MLflow client for later artifact & metric queries ---------- #
    client = MlflowClient(tracking_uri=cfg.mlflow.tracking_uri)

    # record git commit ----------------------------------------------------- #
    try:
        commit_id = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        # ── record in MLflow so the run is fully reproducible ── #
        mlflow.set_tag("git_commit", commit_id)  # searchable tag
        mlflow.log_param("git_commit", commit_id)  # visible in params table
        logging.info("Git commit id: %s", commit_id)

    except Exception as exc:  # pragma: no cover
        logging.warning("Could not fetch git commit: %s", exc)

    # 3 ── Dataset ---------------------------------------------------------
    model_type = cfg.model.type.lower()
    need_marks = model_type in {"fedformer", "informer", "transformer"}

    ds_kwargs: dict[str, Any] = {
        "root_dir": Path(cfg.data.file),
        "seq_len": cfg.data.seq_length,
        "horizon": cfg.data.horizon,
        "single_ticker": True,
        "stride": cfg.data.stride,
        "return_marks": need_marks,
    }

    if need_marks:
        ds_kwargs.update(label_len=cfg.model.label_len, pred_len=cfg.model.pred_len)

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

    # 4 ── Model ------------------------------------------------------------ #
    if model_type == "stocklstm":
        core = StockLSTM(
            input_size=len(full_dataset.feature_cols),
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
        )
        lightning_module: pl.LightningModule = LightningRegressor(
            core, lr=cfg.training.learning_rate
        )

    elif model_type == "transformer":
        lightning_module = LightningTimeSeriesTransformer(
            enc_feat_dim=len(full_dataset.feature_cols),
            projection_dim=cfg.model.projection_dim,
            num_trans_blocks=cfg.model.num_trans_blocks,
            num_heads=cfg.model.num_heads,
            ff_dim=cfg.model.ff_dim,
            mlp_units=cfg.model.mlp_units,
            dropout=cfg.model.dropout,
            lr=cfg.training.learning_rate,
        )

    elif model_type == "fedformer":
        fed_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        lightning_module = LightningFEDformer(
            fed_cfg,
            enc_feat_dim=len(full_dataset.feature_cols),
            learning_rate=cfg.training.learning_rate,
        )

    elif model_type == "informer":
        inf_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        lightning_module = LightningInformer(
            inf_cfg,
            enc_feat_dim=len(full_dataset.feature_cols),
            learning_rate=cfg.training.learning_rate,
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    # 5 ── Trainer & Callbacks -------------------------------------------- #
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

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        logger=mlf_logger,
        callbacks=[ckpt_cb, lr_monitor],
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
    )

    # 6 ── Training --------------------------------------------------------- #
    trainer.fit(lightning_module, train_loader, val_loader)

    # 7 ── Artifacts: best checkpoint -------------------------------------- #
    best_ckpt = ckpt_cb.best_model_path
    mlf_logger.experiment.log_artifact(mlf_logger.run_id, best_ckpt)

    # 8 ── Hydrated config backup ------------------------------------------ #
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # plain Python nested dict

    # Flatten the nested dict so MLflow gets simple key-value pairs
    def _flatten(d, parent_key=""):
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                yield from _flatten(v, new_key)
            else:
                yield new_key, v

    mlf_logger.experiment.log_params(
        mlf_logger.run_id,
        dict(_flatten(cfg_dict)),
    )

    # 9 ── Export metric plots to *plots/* --------------------------------- #
    plots_dir = Path(get_original_cwd()) / "plots"
    metric_keys = [  # ←-- use the real names
        "train_loss_step",  # every batch
        "train_loss_epoch",  # epoch mean (optional)
        "val_loss_epoch",
        "lr-Adam",
    ]
    _fetch_and_plot_metrics(client, mlf_logger.run_id, metric_keys, plots_dir)

    # 10 ── Optional: ONNX export (same as before, unchanged) -------------- #
    try:
        onnx_path = Path(best_ckpt).with_suffix(".onnx")
        if model_type == "stocklstm":
            x, _ = next(iter(train_loader))
            dummy: torch.Tensor = x[:1].cpu()
            lightning_module.to_onnx(
                file_path=onnx_path.as_posix(),
                input_sample=dummy,
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
        nice_name = f"{cfg.model.type}_{Path(cfg.data.file).stem}.onnx"
        shutil.copy2(onnx_path, export_dir / nice_name)
        logging.info("ONNX model saved to %s", export_dir / nice_name)
    except RuntimeError as exc:  # pragma: no cover
        logging.warning("ONNX export skipped: %s", exc)

    logging.info("Training completed ✔")


# --------------------------------------------------------------------------- #
#                                Entry‑point                                 #
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    train()
