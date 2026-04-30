"""
model/transformer.py — Module 3: Temporal Transformer Encoder.

Models the chronological evolution of system state across the sliding window.

Input:  (B, C, D)   — graph-fused node embeddings for a single time step
        (repeated over W time steps in the full pipeline, but for our
         single-snapshot-per-window design, this acts as a sequence
         of C sensor "tokens" with positional encoding)

Output: (B, d_model)  — latent temporal representation
"""

import math
import torch
import torch.nn as nn

from config import NUM_SENSORS, EMBED_DIM, D_MODEL, N_HEADS, NUM_TRANSFORMER_LAYERS, FFN_EXPANSION, DROPOUT


class PositionalEncoding(nn.Module):
    """Standard sine/cosine positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S, d_model)"""
        return x + self.pe[:, :x.size(1)]


class TemporalTransformerEncoder(nn.Module):
    """
    Processes the C sensor embeddings as a sequence (treating sensors as tokens).

    Steps:
        1. Linear projection: C*D → d_model  (or D → d_model per sensor)
        2. Positional Encoding
        3. N Transformer Encoder layers
        4. Mean pooling → (B, d_model)
    """

    def __init__(self, num_sensors: int = NUM_SENSORS, embed_dim: int = EMBED_DIM,
                 d_model: int = D_MODEL, n_heads: int = N_HEADS,
                 num_layers: int = NUM_TRANSFORMER_LAYERS,
                 ffn_expansion: int = FFN_EXPANSION, dropout: float = DROPOUT):
        super().__init__()
        self.num_sensors = num_sensors
        self.d_model = d_model

        # Project each sensor embedding to d_model
        self.input_proj = nn.Linear(embed_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=num_sensors + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ffn_expansion,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D) — graph-fused sensor embeddings

        Returns:
            latent: (B, d_model)
        """
        # Project to transformer dimension
        x = self.input_proj(x)  # (B, C, d_model)

        # Add positional encoding
        x = self.pos_enc(x)

        # Transformer encoding
        x = self.encoder(x)  # (B, C, d_model)
        x = self.norm(x)

        # Mean pooling across sensors
        x = x.mean(dim=1)  # (B, d_model)

        return x
