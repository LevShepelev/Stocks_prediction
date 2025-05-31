import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    """
    A simple LSTM-based regressor for time series data.

    :param input_size: Number of features in each input timestep.
    :param hidden_size: Number of features in the hidden state of the LSTM.
    :param num_layers: Number of stacked LSTM layers.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 50, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LSTM network.

        :param x: Tensor of shape (batch_size, seq_length, input_size).
        :return: Tensor of shape (batch_size, 1) with the regression output.
        """
        lstm_out, _ = self.lstm(x)
        # Take the last hidden state from LSTM and pass through a linear layer
        return self.fc(lstm_out[:, -1, :])
