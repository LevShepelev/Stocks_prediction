# models/lightning_fedformer.py
import pytorch_lightning as pl
import torch.nn as nn
from omegaconf import OmegaConf
from .fed_former import Model as FEDformer
import torch
from types import SimpleNamespace
class LightningFEDformer(pl.LightningModule):
    def __init__(self, fed_cfg_raw, enc_feat_dim: int, learning_rate: float = 1e-3):
        super().__init__()

        # 1️⃣  Resolve and turn *everything* into plain python
        cfg_dict = OmegaConf.to_container(fed_cfg_raw, resolve=True)
        cfg_dict["enc_in"] = cfg_dict["dec_in"] = enc_feat_dim
        cfg_dict["c_out"] = 1
        cfg_dict["output_attention"] = cfg_dict.get("output_attention", False)

        # 🚨 ensure moving_avg is a python *list*, never ListConfig
        if "moving_avg" in cfg_dict and not isinstance(cfg_dict["moving_avg"], list):
            cfg_dict["moving_avg"] = list(cfg_dict["moving_avg"])
        cfg_obj = SimpleNamespace(**cfg_dict)
        # 2️⃣  Re-wrap for attribute-style access expected by FEDformer
        self.net = FEDformer(cfg_obj)
        self.lr = learning_rate
        self.loss = nn.MSELoss()


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.net(x_enc, x_mark_enc, x_dec, x_mark_dec)

    def training_step(self, batch, _):
        x_enc, mark_enc, mark_dec, y = batch
        loss = self._shared_step(x_enc, mark_enc, mark_dec, y)   # helper below
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    # ---------- NEW --------------------------------------------------------
    def validation_step(self, batch, _):
        x_enc, mark_enc, mark_dec, y = batch
        loss = self._shared_step(x_enc, mark_enc, mark_dec, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    # ----------------------------------------------------------------------

    # factor out the common forward–loss code
    def _shared_step(self, x_enc, mark_enc, mark_dec, y):
        # build decoder query
        x_dec = torch.cat(
            [
                x_enc[:, -self.net.label_len :, :],
                torch.zeros_like(x_enc[:, : self.net.pred_len, :]),
            ],
            dim=1,
        )
        out = self(x_enc, mark_enc, x_dec, mark_dec)          # [B, pred_len, 1]
        y_hat_last = out[:, -1, 0]                            # [B]
        return self.loss(y_hat_last, y.squeeze(-1))           # scalar tensor


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
