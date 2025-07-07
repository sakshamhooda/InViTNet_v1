import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ConvTokenizer(nn.Module):
    """Convolutional tokenizer used in Compact Convolutional Transformers.

    Converts the input image into a sequence of embeddings while retaining
    local inductive bias via a convolution front-end followed by pooling.
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 256,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 pool_kernel_size: int = 3, pool_stride: int = 2, pool_padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size, stride, padding)
        self.pool = nn.MaxPool2d(pool_kernel_size, pool_stride, pool_padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)          # (B, D, H, W)
        x = self.pool(x)          # (B, D, H', W')
        x = rearrange(x, 'b d h w -> b (h w) d')  # (B, N, D) sequence of tokens
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (static, not learned)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, seq_len: int, device: torch.device):
        position = torch.arange(seq_len, device=device)[:, None]
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device) *
                             -(math.log(10000.0) / self.d_model))
        pe = torch.zeros(seq_len, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (seq_len, d_model)


class SequencePooling(nn.Module):
    """Attention-based sequence pooling replacing [CLS] token in CCT."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention_pool = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, N, D)
        attn = self.attention_pool(x)  # (B, N, 1)
        attn = torch.softmax(attn, dim=1)  # (B, N, 1)
        pooled = torch.sum(attn * x, dim=1)  # (B, D)
        return pooled


class SpatialCCT(nn.Module):
    """Compact Convolutional Transformer (CCT) backbone with texture weight injection."""

    def __init__(self, num_classes: int = 2, embed_dim: int = 256, depth: int = 7,
                 n_heads: int = 4, mlp_ratio: int = 2, dropout: float = 0.1):
        super().__init__()
        self.tokenizer = ConvTokenizer(in_channels=3, embed_dim=embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads,
                                                   dim_feedforward=embed_dim * mlp_ratio,
                                                   dropout=dropout, activation='gelu',
                                                   batch_first=True)
        self.transformer = nn.ModuleList([encoder_layer for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.seq_pool = SequencePooling(embed_dim)
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor, texture_weights: Optional[torch.Tensor] = None):
        """Forward pass.

        Args:
            x: Image tensor (B, 3, H, W)
            texture_weights: Optional texture weight vector (B, D) used to modulate tokens.
        Returns:
            logits: Classification logits (B, num_classes)
        """
        tokens = self.tokenizer(x)           # (B, N, D)
        B, N, D = tokens.shape
        pos = self.pos_encoding(N, x.device)  # (N, D)
        tokens = tokens + pos.unsqueeze(0)

        # Pass through transformer layers with optional texture injection
        for layer in self.transformer:
            tokens = layer(tokens)
            if texture_weights is not None:
                tokens = tokens * texture_weights.unsqueeze(1)  # broadcast over sequence

        tokens = self.norm(tokens)
        pooled = self.seq_pool(tokens)
        logits = self.fc_out(pooled)
        return logits
