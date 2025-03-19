import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb

from stocks_prediction.dataset import load_all_data
from stocks_prediction.models import StockLSTM, TimeSeriesTransformerWithProjection
from stocks_prediction.visualization import plot_results


# LightningModule wrapper to encapsulate training, validation, and testing steps
class LitWrapper(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and prepare data
    X, y = load_all_data("Data/Stocks", start_date="1990-01-01", seq_length=60)
    if X is None:
        raise ValueError("No data loaded.")

    # Split data into training and validation sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Ensure y has the proper dimensions
    if y_train.dim() == 1:
        y_train = y_train.unsqueeze(-1)
    if y_test.dim() == 1:
        y_test = y_test.unsqueeze(-1)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # --------- LSTM Model Training and Evaluation ---------
    # Initialize wandb logger for LSTM run
    lstm_logger = WandbLogger(
        project="stocks_prediction", name="LSTM_run", log_model=True
    )
    lstm_model = StockLSTM()
    lit_lstm = LitWrapper(lstm_model)
    trainer = Trainer(
        max_epochs=50,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=lstm_logger,
    )
    trainer.fit(lit_lstm, train_loader, val_loader)

    # Evaluate the LSTM model
    lit_lstm.model.eval()
    with torch.no_grad():
        lstm_preds = lit_lstm.model(X_test.to(device)).cpu().numpy()
    y_test_np = y_test.cpu().numpy().squeeze()
    lstm_preds = lstm_preds.squeeze()
    lstm_mse = mean_squared_error(y_test_np, lstm_preds)
    print(f"LSTM Test MSE: {lstm_mse:.4f}")
    wandb.finish()

    # --------- Transformer Model Training and Evaluation ---------
    # Initialize wandb logger for Transformer run
    transformer_logger = WandbLogger(
        project="stocks_prediction", name="Transformer_run", log_model=True
    )
    transformer_model = TimeSeriesTransformerWithProjection(
        projection_dim=4,
        num_trans_blocks=4,
        num_heads=4,
        ff_dim=2,
        mlp_units=[256],
        dropout=0.1,
    )
    lit_transformer = LitWrapper(transformer_model)
    trainer = Trainer(
        max_epochs=2,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=transformer_logger,
    )
    trainer.fit(lit_transformer, train_loader, val_loader)

    # Evaluate the Transformer model
    lit_transformer.model.eval()
    with torch.no_grad():
        transformer_preds = lit_transformer.model(X_test.to(device)).cpu().numpy()
    transformer_mse = mean_squared_error(y_test.cpu().numpy(), transformer_preds)
    print(f"Transformer Test MSE: {transformer_mse:.4f}")
    wandb.finish()

    # Optionally plot results
    plot_results(y_test.cpu().numpy(), lstm_preds)
    plot_results(y_test.cpu().numpy(), transformer_preds)


if __name__ == "__main__":
    main()
