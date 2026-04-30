"""
model/graph_attention.py — Module 2: Phase-Coherence Graph Attention Network.

Fuses sensor representations weighted by physical phase synchronization (PCI).

Input:  node features (B, C, D) + PCI adjacency (B, C, C)
Output: graph-fused node embeddings (B, C, D)

Uses continuous PCI values as soft edge weights rather than binary adjacency,
keeping the spatial module physically interpretable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import EMBED_DIM, N_HEADS


class GATLayer(nn.Module):
    """
    Single Graph Attention layer with PCI-modulated attention.

    For memory efficiency, attention is computed as:
        1. Project nodes to queries/keys/values per head
        2. Compute attention scores (QK^T / sqrt(d_k))
        3. Modulate by PCI adjacency matrix
        4. Softmax + aggregate values
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = N_HEADS,
                 dropout: float = 0.1, concat: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.concat = concat

        if concat:
            assert out_dim % n_heads == 0, "out_dim must be divisible by n_heads when concatenating"
            self.d_k = out_dim // n_heads
        else:
            self.d_k = out_dim

        self.W_q = nn.Linear(in_dim, n_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(in_dim, n_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(in_dim, n_heads * self.d_k, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

        if concat:
            self.out_proj = nn.Linear(n_heads * self.d_k, out_dim)
        else:
            self.out_proj = nn.Linear(self.d_k, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B, C, in_dim)
            adj: (B, C, C)   — PCI adjacency matrix

        Returns:
            out: (B, C, out_dim)
        """
        B, C, _ = x.shape
        H, D = self.n_heads, self.d_k

        # Multi-head projections: (B, C, H, D) → (B, H, C, D)
        Q = self.W_q(x).view(B, C, H, D).permute(0, 2, 1, 3)  # (B, H, C, D)
        K = self.W_k(x).view(B, C, H, D).permute(0, 2, 1, 3)
        V = self.W_v(x).view(B, C, H, D).permute(0, 2, 1, 3)

        # Attention scores: (B, H, C, C)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)

        # Modulate by PCI adjacency — broadcast adj (B, 1, C, C)
        scores = scores * adj.unsqueeze(1)

        # Apply leaky ReLU then softmax
        scores = self.leaky_relu(scores)
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        # Aggregate values: (B, H, C, D)
        out = torch.matmul(attn, V)

        if self.concat:
            # Concatenate heads: (B, C, H*D)
            out = out.permute(0, 2, 1, 3).reshape(B, C, H * D)
        else:
            # Average heads: (B, C, D)
            out = out.mean(dim=1)

        out = self.out_proj(out)
        return out


class PhaseCoherenceGAT(nn.Module):
    """
    Two-layer GAT with PCI-modulated attention.

    GAT Layer 1: Multi-head (8 heads), concat aggregation
    GAT Layer 2: Multi-head (8 heads), concat aggregation + LeakyReLU
    """

    def __init__(self, embed_dim: int = EMBED_DIM, n_heads: int = N_HEADS):
        super().__init__()
        self.gat1 = GATLayer(embed_dim, embed_dim, n_heads=n_heads, concat=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.gat2 = GATLayer(embed_dim, embed_dim, n_heads=n_heads, concat=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B, C, D)
            adj: (B, C, C)

        Returns:
            out: (B, C, D)
        """
        # Layer 1 with residual
        residual = x
        x = self.gat1(x, adj)
        x = self.norm1(x + residual)

        # Layer 2 with residual
        residual = x
        x = self.gat2(x, adj)
        x = self.act(x)
        x = self.norm2(x + residual)

        return x
