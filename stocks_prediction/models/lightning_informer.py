import logging
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf

from .Informer import Model as InformerModel  # reference implementation

logger = logging.getLogger(__name__)


class LightningInformer(pl.LightningModule):
    """PyTorch-Lightning wrapper around the reference Informer model.

    * Converts a Hydra ``DictConfig`` into ``SimpleNamespace`` expected by the
      original implementation.
    * Builds decoder queries on-the-fly
      (``label_len`` most recent steps + zero padding of length ``pred_len``).
    * Accepts batches in two formats:

        1. ``(x, y)`` – produced by datasets **without** calendar marks
        2. ``(x_enc, mark_enc, mark_dec, y)`` – datasets that *do* return marks
    """

    # --------------------------------------------------------------------- #
    #                               Init                                    #
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        inf_cfg_raw: DictConfig,
        enc_feat_dim: int,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["inf_cfg_raw"]
        )  # keep Lightning logs tidy

        # --- resolve Hydra → plain python ---------------------------------
        cfg_dict: Dict[str, Any] = OmegaConf.to_container(
            inf_cfg_raw, resolve=True
        )
        cfg_dict.update(enc_in=enc_feat_dim, dec_in=enc_feat_dim)
        cfg = SimpleNamespace(**cfg_dict)  # the original Informer wants attrs

        # --- build backbone ------------------------------------------------
        self.net = InformerModel(cfg)
        self.loss_fn = nn.MSELoss()
        self.lr: float = learning_rate

        # --- cache frequently used lengths --------------------------------
        # provide sensible fallbacks so the wrapper never crashes on missing
        # attributes
        self.label_len: int = int(getattr(cfg, "label_len", 0))
        self.pred_len: int = int(getattr(cfg, "pred_len", 1))
        self.output_attention: bool = bool(
            getattr(cfg, "output_attention", False)
        )

        logger.debug(
            "Informer wrapper initialised (label_len=%d, pred_len=%d, "
            "output_attention=%s)",
            self.label_len,
            self.pred_len,
            self.output_attention,
        )

    # --------------------------------------------------------------------- #
    #                           helper utilities                             #
    # --------------------------------------------------------------------- #
    def _build_decoder_input(self, x_enc: torch.Tensor) -> torch.Tensor:
        """Last *label_len* steps + zero-padding of length *pred_len*."""
        pad = torch.zeros_like(x_enc[:, : self.pred_len, :])
        return torch.cat((x_enc[:, -self.label_len :, :], pad), dim=1)

    @staticmethod
    def _make_dummy_marks(
        ref: torch.Tensor, pred_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create zero marks when the dataset does not provide them."""
        bsz, enc_len, _ = ref.shape
        device, dtype = ref.device, ref.dtype
        mark_enc = torch.zeros(bsz, enc_len, 4, device=device, dtype=dtype)
        mark_dec = torch.zeros(bsz, pred_len + enc_len, 4, device=device, dtype=dtype)
        return mark_enc, mark_dec

    # --------------------------------------------------------------------- #
    #                        forward & PL hooks                              #
    # --------------------------------------------------------------------- #
    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
    ) -> torch.Tensor:
        """Delegate to Informer implementation; keep signature untouched."""
        return self.net(x_enc, x_mark_enc, x_dec, x_mark_dec)

    # ─── shared logic for train / val ──────────────────────────────────────
    def _shared_step(
        self, batch: Tuple[torch.Tensor, ...], stage: str
    ) -> torch.Tensor:
        try:
            if len(batch) == 4:
                # full tuple with calendar marks
                x_enc, mark_enc, mark_dec, y = batch
            elif len(batch) == 2:
                # dataset without marks → fabricate dummy zeros
                x_enc, y = batch
                mark_enc, mark_dec = self._make_dummy_marks(
                    x_enc, self.pred_len
                )
            else:
                raise ValueError(
                    f"Unexpected batch format of length {len(batch)}"
                )

            x_dec = self._build_decoder_input(x_enc)

            out = self(x_enc, mark_enc, x_dec, mark_dec)
            y_hat = out[0] if self.output_attention else out  # [B, T, 1]

            # y may be [B, 1] (single horizon) or [B, T, 1]
            if y.ndim == 2:
                y_true = y.squeeze(-1)          # [B]
                y_pred = y_hat[:, -1, 0]        # last step only
            else:
                y_true = y[:, -self.pred_len :, :]
                y_pred = y_hat

            loss = self.loss_fn(y_pred, y_true)
            self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
            return loss

        except Exception as exc:  # noqa: BLE001
            # fail fast but add context for easier debugging
            logger.exception("Error in %s step: %s", stage, exc)
            raise

    # ─── Lightning API ─────────────────────────────────────────────────────
    def training_step(self, batch: Tuple[torch.Tensor, ...], _: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Tuple[torch.Tensor, ...], _: int) -> None:
        _ = self._shared_step(batch, "val")

    # --------------------------------------------------------------------- #
    #                       optimiser / scheduler                            #
    # --------------------------------------------------------------------- #
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
