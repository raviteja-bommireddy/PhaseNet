"""
model/phasenet.py — Full PhaseNet++ model (end-to-end assembly).

Assembles all four modules into the complete autoencoder:
    Module 1: Time-Frequency Vision Embedding (CNN)
    Module 2: Phase-Coherence Graph Attention Network (GAT)
    Module 3: Temporal Transformer Encoder
    Module 4: Dual-Head Decoder (Magnitude + Phase)
"""

import torch
import torch.nn as nn

from config import NUM_SENSORS, FREQ_BINS, EMBED_DIM, D_MODEL, N_HEADS
from model.embedding import TimeFrequencyEmbedding
from model.graph_attention import PhaseCoherenceGAT
from model.transformer import TemporalTransformerEncoder
from model.decoder import DualHeadDecoder


class PhaseNetPP(nn.Module):
    """
    PhaseNet++ — Phase-Aware Spatio-Temporal Autoencoder for
    Cyber Attack Detection in Multivariate Sensor Systems.

    Forward pass:
        Input:  STFT tensor (B, C, 2, F) + PCI matrix (B, C, C)
        Output: mag_hat (B, C, F), phase_hat (B, C, F)
    """

    def __init__(self, num_sensors: int = NUM_SENSORS, freq_bins: int = FREQ_BINS,
                 embed_dim: int = EMBED_DIM, d_model: int = D_MODEL,
                 n_heads: int = N_HEADS):
        super().__init__()

        # Module 1 — CNN spectral embedding
        self.embedding = TimeFrequencyEmbedding(
            freq_bins=freq_bins,
            embed_dim=embed_dim,
        )

        # Module 2 — PCI-weighted GAT
        self.gat = PhaseCoherenceGAT(
            embed_dim=embed_dim,
            n_heads=n_heads,
        )

        # Module 3 — Temporal Transformer Encoder
        self.transformer = TemporalTransformerEncoder(
            num_sensors=num_sensors,
            embed_dim=embed_dim,
            d_model=d_model,
            n_heads=n_heads,
        )

        # Module 4 — Dual-Head Decoder
        self.decoder = DualHeadDecoder(
            d_model=d_model,
            num_sensors=num_sensors,
            freq_bins=freq_bins,
        )

    def forward(self, stft: torch.Tensor, pci: torch.Tensor) -> tuple:
        """
        Args:
            stft: (B, C, 2, F) — STFT magnitude + phase per sensor
            pci:  (B, C, C)    — phase coherence adjacency matrix

        Returns:
            mag_hat:   (B, C, F) — reconstructed magnitude
            phase_hat: (B, C, F) — reconstructed phase
        """
        # Module 1: Extract per-sensor spectral embeddings
        node_embeddings = self.embedding(stft)  # (B, C, D)

        # Module 2: Graph attention fusion with PCI weights
        fused_embeddings = self.gat(node_embeddings, pci)  # (B, C, D)

        # Module 3: Temporal transformer encoding
        latent = self.transformer(fused_embeddings)  # (B, d_model)

        # Module 4: Dual-head reconstruction
        mag_hat, phase_hat = self.decoder(latent)  # (B, C, F), (B, C, F)

        return mag_hat, phase_hat

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
