import torch
from mlops_project.models import StockLSTM


def test_stock_lstm_forward():
    model = StockLSTM()
    # Create a dummy input: batch size 10, sequence length 60, 1 feature
    dummy_input = torch.randn(10, 60, 1)
    output = model(dummy_input)
    # Assert that the output has the expected shape: (batch_size, 1)
    assert output.shape == (10, 1)
