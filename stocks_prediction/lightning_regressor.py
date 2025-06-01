"""Generic Lightning wrapper for regression models."""

from __future__ import annotations

import logging
from typing import Tuple

import pytorch_lightning as pl
import torch
from torch import nn, optim


logger = logging.getLogger(__name__)


class LightningRegressor(pl.LightningModule):
    """Wrap any ``nn.Module`` that returns a 1-D regression output."""

    def __init__(self, model: nn.Module, lr: float = 1e-3) -> None:
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])
        self.criterion = nn.MSELoss()

    # ------------------------------------------------------------------ #
    #                            Forward / Steps                         #
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.model(x)

    def _shared_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self(x).squeeze(-1)
        loss = self.criterion(y_hat, y.squeeze(-1))
        self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], _
    ) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> None:
        self._shared_step(batch, "val")

    # ------------------------------------------------------------------ #
    #                          Optimiser / Scheduler                     #
    # ------------------------------------------------------------------ #
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
