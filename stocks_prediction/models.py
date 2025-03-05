import torch.nn as nn


class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


class TransformerBlock(nn.Module):
    """
    A single Transformer encoder block.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.conv1 = nn.Conv1d(
            in_channels=embed_dim, out_channels=ff_dim, kernel_size=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=ff_dim, out_channels=embed_dim, kernel_size=1
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        x_ff = x.transpose(1, 2)
        x_ff = self.conv1(x_ff)
        x_ff = nn.functional.relu(x_ff)
        x_ff = self.dropout2(x_ff)
        x_ff = self.conv2(x_ff)
        x_ff = x_ff.transpose(1, 2)

        x = x + x_ff
        x = self.norm2(x)
        return x


class TimeSeriesTransformer(nn.Module):
    """
    Stacks multiple TransformerBlocks,
    applies global pooling, and an MLP for regression.
    """

    def __init__(
        self,
        num_trans_blocks=4,
        embed_dim=4,  # Must match the projected input feature dimension
        num_heads=4,
        ff_dim=2,
        mlp_units=[256],
        dropout=0.1,
    ):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
                for _ in range(num_trans_blocks)
            ]
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        mlp_layers = []
        in_features = embed_dim
        for units in mlp_units:
            mlp_layers.append(nn.Linear(in_features, units))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            in_features = units
        mlp_layers.append(nn.Linear(in_features, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        for block in self.transformer_blocks:
            x = block(x)
        x = x.transpose(1, 2)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        return self.mlp(x)


class TimeSeriesTransformerWithProjection(nn.Module):
    """
    Projects 1-dimensional input data into a higher-dimensional
    space before feeding it to the Transformer.
    """

    def __init__(self, projection_dim=4, **kwargs):
        super().__init__()
        self.projection = nn.Linear(1, projection_dim)
        self.transformer = TimeSeriesTransformer(embed_dim=projection_dim, **kwargs)

    def forward(self, x):
        x = self.projection(x)
        return self.transformer(x)
