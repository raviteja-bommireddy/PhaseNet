"""
model/decoder.py — Module 4: Dual-Head Magnitude + Phase Decoder.

Reconstructs both the magnitude and phase spectra from the latent representation.

Head A — Magnitude:  Linear(d_model → C×F) → Reshape → ReLU
Head B — Phase:      Linear(d_model → C×F) → Reshape → Tanh × π

Input:   (B, d_model)
Output:  mag_hat (B, C, F),  phase_hat (B, C, F)
"""

import math
import torch
import torch.nn as nn

from config import NUM_SENSORS, FREQ_BINS, D_MODEL


class DualHeadDecoder(nn.Module):
    """
    Dual-head decoder: separate branches for magnitude and phase reconstruction.

    • Magnitude head uses ReLU to enforce non-negativity.
    • Phase head uses Tanh × π to bound output to [−π, π].
    """

    def __init__(self, d_model: int = D_MODEL, num_sensors: int = NUM_SENSORS,
                 freq_bins: int = FREQ_BINS):
        super().__init__()
        self.num_sensors = num_sensors
        self.freq_bins = freq_bins
        out_features = num_sensors * freq_bins

        # Magnitude decoder
        self.mag_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_features),
        )

        # Phase decoder
        self.phase_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_features),
        )

    def forward(self, z: torch.Tensor) -> tuple:
        """
        Args:
            z: (B, d_model) latent representation

        Returns:
            mag_hat:   (B, C, F) — reconstructed magnitudes (non-negative)
            phase_hat: (B, C, F) — reconstructed phases in [−π, π]
        """
        B = z.size(0)

        # Magnitude: ReLU ensures non-negativity
        mag_hat = self.mag_decoder(z)
        mag_hat = mag_hat.view(B, self.num_sensors, self.freq_bins)
        mag_hat = torch.relu(mag_hat)

        # Phase: Tanh × π bounds to [−π, π]
        phase_hat = self.phase_decoder(z)
        phase_hat = phase_hat.view(B, self.num_sensors, self.freq_bins)
        phase_hat = torch.tanh(phase_hat) * math.pi

        return mag_hat, phase_hat
