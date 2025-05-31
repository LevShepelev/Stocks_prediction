from typing import List

import torch
import torch.nn as nn

class MultiHeadSelfAttentionBlock(nn.Module):
    """
    Sub-block that performs multi-head self-attention with a residual connection.

    :param embed_dim: Dimensionality of each token embedding (input size to attention).
    :param num_heads: Number of attention heads.
    :param dropout: Dropout probability.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multi-head self-attention block.

        :param x: Tensor of shape (batch_size, seq_length, embed_dim).
        :return: Tensor of the same shape after attention + residual + normalization.
        """
        skip = x
        attn_out, _ = self.attention(x, x, x)
        x = skip + self.dropout1(attn_out)
        x = self.norm1(x)
        return x


class FeedForwardBlock(nn.Module):
    """
    Position-wise feed-forward sub-block with a residual connection.

    :param embed_dim: Dimensionality of each token embedding.
    :param ff_dim: Hidden size of the intermediate dense (conv1d) layer.
    :param dropout: Dropout probability.
    """

    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=embed_dim, out_channels=ff_dim, kernel_size=1
        )
        self.dropout2 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            in_channels=ff_dim, out_channels=embed_dim, kernel_size=1
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feed-forward block.

        :param x: Tensor of shape (batch_size, seq_length, embed_dim).
        :return: Tensor of the same shape after feed-forward + residual + normalization.
        """
        skip = x
        # Switch to (batch_size, embed_dim, seq_length) for conv
        out = x.transpose(1, 2)
        out = self.conv1(out)
        out = nn.functional.relu(out)
        out = self.dropout2(out)
        out = self.conv2(out)
        # Switch back to (batch_size, seq_length, embed_dim)
        out = out.transpose(1, 2)
        x = skip + out
        x = self.norm2(x)
        return x


class TransformerBlock(nn.Sequential):
    """
    A single Transformer encoder block, composed of multi-head self-attention
    and a feed-forward sub-block (each with its own residual connection).

    :param embed_dim: Dimensionality of each token embedding.
    :param num_heads: Number of attention heads in the self-attention sub-block.
    :param ff_dim: Hidden size used in the feed-forward sub-block.
    :param dropout: Dropout probability.
    """

    def __init__(
        self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1
    ):
        super().__init__(
            MultiHeadSelfAttentionBlock(embed_dim, num_heads, dropout),
            FeedForwardBlock(embed_dim, ff_dim, dropout),
        )


class TimeSeriesTransformer(nn.Module):
    """
    A Transformer-based model for time series regression.

    It stacks multiple Transformer encoder blocks, applies a global average pool,
    and uses an MLP to produce a final regression output.

    :param num_trans_blocks: Number of stacked TransformerBlocks.
    :param embed_dim: Dimensionality of each token embedding.
    :param num_heads: Number of attention heads in each block.
    :param ff_dim: Hidden size used in the feed-forward sub-block of each block.
    :param mlp_units: List of hidden sizes for the final MLP layers.
    :param dropout: Dropout probability for attention and feed-forward layers.
    """

    def __init__(
        self,
        num_trans_blocks: int = 4,
        embed_dim: int = 4,
        num_heads: int = 4,
        ff_dim: int = 2,
        mlp_units: List[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if mlp_units is None:
            mlp_units = [256]

        # Build the Transformer encoder blocks
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
                for _ in range(num_trans_blocks)
            ]
        )

        # Global pooling to reduce the sequence dimension to 1
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Build the MLP for final regression
        mlp_layers = []
        in_features = embed_dim
        for units in mlp_units:
            mlp_layers.append(nn.Linear(in_features, units))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            in_features = units
        mlp_layers.append(nn.Linear(in_features, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer-based time series model.

        :param x: Tensor of shape (batch_size, seq_length, embed_dim).
        :return: Tensor of shape (batch_size, 1) with regression predictions.
        """
        x = self.transformer_blocks(x)
        # Switch shape for pooling: (batch_size, embed_dim, seq_length)
        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        return self.mlp(x)


class TimeSeriesTransformerWithProjection(nn.Module):
    """
    A variant of TimeSeriesTransformer that first projects 1D input
    into a higher dimensional embedding before feeding it to the Transformer.

    :param projection_dim: Dimensionality of the projection layer output.
    :param kwargs: Additional parameters for TimeSeriesTransformer (e.g. num_trans_blocks, embed_dim, etc.).
    """

    def __init__(self, projection_dim: int = 4, **kwargs):
        super().__init__()
        self.projection = nn.Linear(1, projection_dim)
        self.transformer = TimeSeriesTransformer(embed_dim=projection_dim, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the projected Transformer.

        :param x: Tensor of shape (batch_size, seq_length, 1).
        :return: Tensor of shape (batch_size, 1) with regression predictions.
        """
        x = self.projection(x)
        return self.transformer(x)
