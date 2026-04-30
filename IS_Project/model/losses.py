"""
model/losses.py — Magnitude, phase (circular), and coherence losses.

Three-term composite loss:
    L_total = α·MSE(|Ẑ|, |Z|) + β·(1 − cos(φ̂ − φ)) + γ·MSE(PCI_hat, PCI)

The circular phase loss handles ±π wrapping correctly.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ALPHA, BETA, GAMMA, FREQ_BINS


class PhaseNetLoss(nn.Module):
    """
    Composite loss for PhaseNet++:
        magnitude_loss  = MSE(|Ẑ|, |Z|)
        phase_loss      = mean(1 − cos(φ̂ − φ))
        coherence_loss  = MSE(PCI_reconstructed, PCI_original)

    The coherence term uses a differentiable PCI re-estimate from the
    predicted phases, creating a gradient path that encourages the model
    to preserve inter-sensor synchronization.
    """

    def __init__(self, alpha: float = ALPHA, beta: float = BETA,
                 gamma: float = GAMMA):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def _compute_pci_differentiable(self, phase: torch.Tensor) -> torch.Tensor:
        """
        Differentiable PCI computation from predicted phases.

        Args:
            phase: (B, C, F) — reconstructed phase values

        Returns:
            pci: (B, C, C) — reconstructed PCI matrix
        """
        # phase_i - phase_j for all pairs
        # (B, C, 1, F) - (B, 1, C, F) → (B, C, C, F)
        diff = phase.unsqueeze(2) - phase.unsqueeze(1)

        # PCI = |mean(exp(i * diff))|
        # Use cos and sin to keep it differentiable
        real = torch.cos(diff).mean(dim=-1)  # (B, C, C)
        imag = torch.sin(diff).mean(dim=-1)  # (B, C, C)
        pci = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)  # (B, C, C)

        return pci

    def forward(self, mag_hat, phase_hat, mag_target, phase_target, pci_target):
        """
        Args:
            mag_hat:      (B, C, F) — predicted magnitude
            phase_hat:    (B, C, F) — predicted phase
            mag_target:   (B, C, F) — ground truth magnitude
            phase_target: (B, C, F) — ground truth phase
            pci_target:   (B, C, C) — ground truth PCI matrix

        Returns:
            total_loss, (mag_loss, phase_loss, coh_loss)
        """
        # 1. Magnitude MSE loss
        mag_loss = F.mse_loss(mag_hat, mag_target)

        # 2. Circular phase loss: mean(1 - cos(φ̂ - φ))
        phase_diff = phase_hat - phase_target
        phase_loss = (1.0 - torch.cos(phase_diff)).mean()

        # 3. Coherence loss: MSE between reconstructed and original PCI
        pci_hat = self._compute_pci_differentiable(phase_hat)
        coh_loss = F.mse_loss(pci_hat, pci_target)

        # Composite
        total = self.alpha * mag_loss + self.beta * phase_loss + self.gamma * coh_loss

        return total, (mag_loss.item(), phase_loss.item(), coh_loss.item())
