import logging

# Import the refactored data loading code
from pathlib import Path

import hydra
import mlflow
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from dataset import StockDataset, load_all_data  # <-- import here

# Import your models
from models import StockLSTM, TimeSeriesTransformerWithProjection


class LightningRegressor(pl.LightningModule):
    """
    A generic LightningModule wrapper that trains a given PyTorch model
    for regression using MSE loss and Adam optimizer.
    """

    def __init__(self, model: nn.Module, learning_rate: float = 1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # lightning module
    def training_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return None  # don’t hand the tensor back to PL

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        predictions = self(features)
        loss = self.criterion(predictions, labels)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)


@hydra.main(version_base=None, config_path=".", config_name="train_config")
def train(cfg: DictConfig):
    """
    Train function using Hydra for config management,
    PyTorch Lightning for training, and MLflow for experiment tracking.

    This function can train either StockLSTM or TimeSeriesTransformerWithProjection
    based on the config: cfg.model.type = "lstm" or "transformer".
    """

    # 1. Logging Setup
    logging.basicConfig(level=getattr(logging, cfg.Logging.level))
    logging.info("Hydra Config:\n" + OmegaConf.to_yaml(cfg))

    # 2. MLflow Setup
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # 3. Load Data
    try:
        all_pairs = load_all_data(
            directory=Path(cfg.data.directory),
            start_date=cfg.data.start_date,
            seq_length=cfg.data.seq_length,
            log_info=cfg.data.log_info,
        )
    except ValueError as e:
        logging.error(f"Failed to load data: {e}")
        return

    full_dataset = StockDataset(all_pairs)

    # Optional: train/val split
    train_size = int(len(full_dataset) * cfg.training.train_val_split)
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=0,  # test with a single worker first
        pin_memory=True,
        persistent_workers=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )

    # 4. Model Creation
    model_type = cfg.model.type.lower()
    if model_type == "lstm":
        model_instance = StockLSTM(
            input_size=cfg.model.input_size,
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
        )
    elif model_type == "transformer":
        model_instance = TimeSeriesTransformerWithProjection(
            projection_dim=cfg.model.projection_dim,
            num_trans_blocks=cfg.model.num_trans_blocks,
            num_heads=cfg.model.num_heads,
            ff_dim=cfg.model.ff_dim,
            mlp_units=cfg.model.mlp_units,
            dropout=cfg.model.dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    lightning_module = LightningRegressor(
        model=model_instance, learning_rate=cfg.training.learning_rate
    )

    # 5. PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
    )

    # 6. MLflow Logging
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(
            {
                "model_type": cfg.model.type,
                "input_size": cfg.model.input_size,
                "hidden_size": cfg.model.hidden_size,
                "num_layers": cfg.model.num_layers,
                "learning_rate": cfg.training.learning_rate,
                "batch_size": cfg.training.batch_size,
                "epochs": cfg.training.epochs,
            }
        )

        if model_type == "transformer":
            mlflow.log_params(
                {
                    "projection_dim": cfg.model.projection_dim,
                    "num_trans_blocks": cfg.model.num_trans_blocks,
                    "num_heads": cfg.model.num_heads,
                    "ff_dim": cfg.model.ff_dim,
                    "mlp_units": cfg.model.mlp_units,
                    "dropout": cfg.model.dropout,
                }
            )

        trainer.fit(lightning_module, train_loader, val_loader)

        # Log the final model checkpoint
        mlflow.pytorch.log_model(
            pytorch_model=lightning_module, artifact_path="models/lightning_regressor"
        )

    logging.info("Training completed successfully.")


if __name__ == "__main__":
    train()
