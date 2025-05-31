import pytorch_lightning as pl
import torch
from torch import nn, optim
from typing import List, Tuple, Union

from stocks_prediction.models.time_series_transformer_with_projection import TimeSeriesTransformer


class LightningTimeSeriesTransformer(pl.LightningModule):
    """
    LightningModule wrapper for TimeSeriesTransformer with explicit projection from encoder feature dimension.

    Handles both simple (x, y) and complex (x_enc, mark_enc, mark_dec, y) batch formats,
    and supports ONNX export by accepting multiple inputs.
    """

    def __init__(
        self,
        enc_feat_dim: int,
        projection_dim: int = 4,
        num_trans_blocks: int = 4,
        num_heads: int = 4,
        ff_dim: int = 2,
        mlp_units: List[int] = None,
        dropout: float = 0.1,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        # Save all init args to hparams for logging/checkpointing
        self.save_hyperparameters()

        # Projection layer: map input features to embedding dimension
        self.projection = nn.Linear(
            self.hparams.enc_feat_dim, self.hparams.projection_dim
        )
        # Core Transformer model
        self.transformer = TimeSeriesTransformer(
            num_trans_blocks=self.hparams.num_trans_blocks,
            embed_dim=self.hparams.projection_dim,
            num_heads=self.hparams.num_heads,
            ff_dim=self.hparams.ff_dim,
            mlp_units=self.hparams.mlp_units or [256],
            dropout=self.hparams.dropout,
        )
        self.criterion = nn.MSELoss()

    def forward(self, *args) -> torch.Tensor:
        """
        Forward pass, supporting:
        - raw inputs: forward(x)
        - dummy tuple for ONNX: forward(x_enc, mark_enc, x_dec, mark_dec)

        Extracts the first tensor as x.
        """
        x = args[0]
        proj = self.projection(x)
        return self.transformer(proj)

    def _step(self, batch: Union[Tuple, List], stage: str) -> torch.Tensor:
        """
        Shared train/val logic: unpack batch, compute MSE loss, log metric.
        """
        # Unpack batch formats
        if len(batch) == 2:
            x, y = batch
        else:
            x_enc, mark_enc, mark_dec, y = batch  # ignore marks
            x = x_enc
        y_hat = self(projection(x) if False else x).squeeze(-1)
        # Actually call forward through projection + transformer
        # But using self(...) goes through our forward
        y_hat = self(x).squeeze(-1)
        loss = self.criterion(y_hat, y.squeeze(-1))
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch: Union[Tuple, List], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Union[Tuple, List], batch_idx: int) -> None:
        self._step(batch, "val")

    def configure_optimizers(self) -> optim.Optimizer:
        """
        Configure Adam optimizer with learning rate.
        """
        return optim.Adam(self.parameters(), lr=self.hparams.lr)
