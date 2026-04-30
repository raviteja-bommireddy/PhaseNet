"""
baselines/models.py — Comparison baselines (LSTM-AE, vanilla Transformer-AE, etc.)

These simpler architectures serve as ablation comparisons to show
the value of PhaseNet++'s phase-aware design.
"""

import torch
import torch.nn as nn
import math

from config import NUM_SENSORS, FREQ_BINS, D_MODEL, DROPOUT


class LSTMAutoencoder(nn.Module):
    """
    LSTM-based autoencoder baseline.
    Encodes sensor magnitude+phase as a flat vector per window,
    processes through bidirectional LSTM, then decodes.
    """

    def __init__(self, num_sensors: int = NUM_SENSORS, freq_bins: int = FREQ_BINS,
                 hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.num_sensors = num_sensors
        self.freq_bins = freq_bins
        input_dim = num_sensors * 2 * freq_bins  # flattened STFT

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT,
        )
        self.bottleneck = nn.Linear(hidden_dim * 2, hidden_dim)

        self.mag_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, num_sensors * freq_bins),
        )
        self.phase_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, num_sensors * freq_bins),
        )

    def forward(self, stft: torch.Tensor, pci: torch.Tensor = None) -> tuple:
        """
        Args:
            stft: (B, C, 2, F)
            pci:  ignored (no graph component)
        """
        B = stft.size(0)
        x = stft.reshape(B, 1, -1)  # (B, 1, C*2*F)

        _, (h_n, _) = self.encoder(x)
        # Concatenate forward and backward
        h = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (B, hidden*2)
        z = self.bottleneck(h)  # (B, hidden)

        mag_hat = torch.relu(self.mag_decoder(z).view(B, self.num_sensors, self.freq_bins))
        phase_hat = torch.tanh(self.phase_decoder(z).view(B, self.num_sensors, self.freq_bins)) * math.pi

        return mag_hat, phase_hat


class VanillaTransformerAE(nn.Module):
    """
    Vanilla Transformer autoencoder baseline — no phase coherence, no GAT.
    """

    def __init__(self, num_sensors: int = NUM_SENSORS, freq_bins: int = FREQ_BINS,
                 d_model: int = D_MODEL, n_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.num_sensors = num_sensors
        self.freq_bins = freq_bins
        input_dim = 2 * freq_bins  # per sensor: magnitude + phase

        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=DROPOUT,
            activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mag_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_sensors * freq_bins),
        )
        self.phase_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_sensors * freq_bins),
        )

    def forward(self, stft: torch.Tensor, pci: torch.Tensor = None) -> tuple:
        B, C, _, F = stft.shape
        x = stft.reshape(B, C, -1)  # (B, C, 2*F)
        x = self.input_proj(x)       # (B, C, d_model)
        x = self.encoder(x)          # (B, C, d_model)
        z = x.mean(dim=1)            # (B, d_model)

        mag_hat = torch.relu(self.mag_decoder(z).view(B, self.num_sensors, self.freq_bins))
        phase_hat = torch.tanh(self.phase_decoder(z).view(B, self.num_sensors, self.freq_bins)) * math.pi

        return mag_hat, phase_hat
