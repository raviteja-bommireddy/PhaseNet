"""
model/embedding.py — Module 1: Time-Frequency Vision Embedding (CNN).

Extracts local spectral texture per sensor, independently.

Input:   (B, C, 2, F)  — magnitude + phase per sensor per freq bin
Output:  (B, C, D)     — one D-dimensional embedding per sensor
"""

import torch
import torch.nn as nn

from config import FREQ_BINS, EMBED_DIM


class TimeFrequencyEmbedding(nn.Module):
    """
    CNN-based spectral feature extractor applied independently per sensor.

    Architecture:
        Conv1D(2→16, k=3, pad=1) → BN → GeLU
        Conv1D(16→32, k=3, pad=1) → BN → GeLU
        MaxPool1D(k=2)
        Flatten + Linear → D
    """

    def __init__(self, freq_bins: int = FREQ_BINS, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.freq_bins = freq_bins
        self.embed_dim = embed_dim

        self.conv = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # After MaxPool: F//2 frequency bins
        pooled_F = freq_bins // 2
        self.fc = nn.Linear(32 * pooled_F, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, 2, F)

        Returns:
            embeddings: (B, C, D)
        """
        B, C, _, F = x.shape

        # Reshape to apply CNN independently per sensor: (B*C, 2, F)
        x = x.reshape(B * C, 2, F)

        # Conv layers
        x = self.conv(x)  # (B*C, 32, F//2)

        # Flatten and project
        x = x.reshape(B * C, -1)  # (B*C, 32 * F//2)
        x = self.fc(x)             # (B*C, D)

        return x.reshape(B, C, self.embed_dim)  # (B, C, D)
