"""
preprocessing/stft_pci.py — STFT computation and PCI matrix generation.

Converts raw sensor windows into:
  • STFT tensor  X  of shape (W, C, 2, F)   — magnitude & phase per sensor per freq bin
  • PCI matrix   A  of shape (W, C, C)       — phase coherence between all sensor pairs

All computation is done in NumPy/SciPy on CPU to keep the GPU pipeline
bottleneck-free (as specified in the README).
"""

import numpy as np
from scipy.signal import stft as scipy_stft


def compute_stft(window: np.ndarray, n_fft: int = 128) -> tuple:
    """
    Compute the Short-Time Fourier Transform for each sensor in a window.

    Args:
        window: (W, C) array — W time steps, C sensors.
        n_fft:  FFT length.

    Returns:
        magnitude: (C, F) array
        phase:     (C, F) array   values in [-π, π]
    """
    W, C = window.shape
    F = n_fft // 2 + 1

    # scipy.signal.stft with axis=0 operates over the time axis for all sensors simultaneously
    _, _, Zxx = scipy_stft(
        window,
        nperseg=min(n_fft, W),
        noverlap=min(n_fft, W) - 1,
        nfft=n_fft,
        boundary=None,
        padded=False,
        axis=0,
    )
    
    # Zxx shape becomes (F, C, 1) when time frames collapse to 1
    # Average across the single time frame to preserve identical dimension logic
    Zxx_mean = Zxx.mean(axis=-1)  # (F, C)
    
    magnitudes = np.abs(Zxx_mean).astype(np.float32).T  # (C, F)
    phases = np.angle(Zxx_mean).astype(np.float32).T    # (C, F)

    return magnitudes, phases


def compute_pci(phases: np.ndarray) -> np.ndarray:
    """
    Compute the Phase Coherence Index matrix for all sensor pairs.

    PCI_AB = | (1/F) * Σ_f exp(i*(φ_A(f) − φ_B(f))) |

    Args:
        phases: (C, F) array of phase angles.

    Returns:
        pci: (C, C) symmetric matrix with values in [0, 1].
    """
    C, F = phases.shape
    # Compute phase differences for all pairs efficiently
    # phases[i] - phases[j] for all i, j
    diff = phases[:, np.newaxis, :] - phases[np.newaxis, :, :]  # (C, C, F)
    # Mean of complex exponentials
    pci = np.abs(np.mean(np.exp(1j * diff), axis=2)).astype(np.float32)  # (C, C)
    return pci


def preprocess_window(window: np.ndarray, n_fft: int = 128) -> tuple:
    """
    Full preprocessing for a single window.

    Args:
        window: (W, C) array.
        n_fft:  FFT length.

    Returns:
        stft_tensor: (C, 2, F) — channel 0 = magnitude, channel 1 = phase
        pci_matrix:  (C, C)
    """
    mag, phase = compute_stft(window, n_fft=n_fft)
    pci = compute_pci(phase)
    # Stack magnitude and phase → (C, 2, F)
    stft_tensor = np.stack([mag, phase], axis=1)
    return stft_tensor, pci
