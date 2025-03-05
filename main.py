import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from stocks_prediction.dataset import load_all_data
from stocks_prediction.models import StockLSTM, TimeSeriesTransformerWithProjection
from stocks_prediction.train_loop import train_model_with_loader
from stocks_prediction.visualization import plot_results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your data (X and y from load_all_data should now have proper shapes)
X, y = load_all_data("Stocks", start_date="1990-01-01", seq_length=60)
if X is None:
    raise ValueError("No data loaded.")

# Split the data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Check and adjust y dimensions if necessary
if y_train.dim() == 1:
    y_train = y_train.unsqueeze(-1)
if y_test.dim() == 1:
    y_test = y_test.unsqueeze(-1)

# Prepare DataLoader for training
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Train LSTM Model
lstm_model = StockLSTM()
lstm_model = train_model_with_loader(
    lstm_model, train_loader, epochs=50, learning_rate=1e-3, device=device
)
# Evaluate the LSTM model
lstm_model.eval()
with torch.no_grad():
    lstm_preds = lstm_model(X_test.to(device)).cpu().numpy()

# Remove any extra dimensions: convert shape from (N,1,1) to (N,) or (N,1)
y_test_np = y_test.cpu().numpy().squeeze()
lstm_preds = lstm_preds.squeeze()

lstm_mse = mean_squared_error(y_test_np, lstm_preds)
print(f"LSTM Test MSE: {lstm_mse:.4f}")


# Train Transformer Model
transformer_model = TimeSeriesTransformerWithProjection(
    projection_dim=4,
    num_trans_blocks=4,
    num_heads=4,
    ff_dim=2,
    mlp_units=[256],
    dropout=0.1,
)
transformer_model = train_model_with_loader(
    transformer_model, train_loader, epochs=50, learning_rate=1e-3, device=device
)
transformer_model.eval()
with torch.no_grad():
    transformer_preds = transformer_model(X_test.to(device)).cpu().numpy()
transformer_mse = mean_squared_error(y_test.cpu().numpy(), transformer_preds)
print(f"Transformer Test MSE: {transformer_mse:.4f}")

# Optionally plot results
plot_results(y_test.cpu().numpy(), lstm_preds)
plot_results(y_test.cpu().numpy(), transformer_preds)
