"""
inference.py — Inference and anomaly scoring.

Loads a trained PhaseNet++ checkpoint and computes per-window anomaly scores
(L_total) on any dataset. The anomaly threshold τ₀.₉₆ is derived from the
96th percentile of validation-set scores.
"""

import os
import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from config import (
    DEVICE, CHECKPOINT_PATH, OUTPUT_DIR, THRESHOLD_PERCENTILE,
)
from model.phasenet import PhaseNetPP
from model.losses import PhaseNetLoss


def load_model(checkpoint_path: str = CHECKPOINT_PATH) -> PhaseNetPP:
    """Load model from checkpoint."""
    model = PhaseNetPP().to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Loaded model from {checkpoint_path} (epoch {checkpoint['epoch']})")
    return model


@torch.no_grad()
def compute_anomaly_scores(model, loader, criterion=None):
    """
    Compute per-window anomaly scores.

    Returns:
        scores:      (N,) array of total reconstruction loss per window
        mag_scores:  (N,) magnitude reconstruction error
        phase_scores:(N,) phase reconstruction error
        coh_scores:  (N,) coherence reconstruction error
        labels:      (N,) labels if available, else None
    """
    if criterion is None:
        criterion = PhaseNetLoss()

    model.eval()
    all_scores = []
    all_mag = []
    all_phase = []
    all_coh = []
    all_labels = []

    for batch in tqdm(loader, desc="  Scoring", leave=False):
        has_labels = len(batch) == 3
        stft = batch[0].to(DEVICE)
        pci = batch[1].to(DEVICE)

        mag_target = stft[:, :, 0, :]
        phase_target = stft[:, :, 1, :]

        with autocast():
            mag_hat, phase_hat = model(stft, pci)

        # Compute per-sample losses (not batch-mean)
        B = stft.size(0)
        for i in range(B):
            loss, (ml, pl, cl) = criterion(
                mag_hat[i:i+1], phase_hat[i:i+1],
                mag_target[i:i+1], phase_target[i:i+1],
                pci[i:i+1]
            )
            all_scores.append(loss.item())
            all_mag.append(ml)
            all_phase.append(pl)
            all_coh.append(cl)

        if has_labels:
            all_labels.extend(batch[2].numpy().tolist())

    scores = np.array(all_scores)
    mag_scores = np.array(all_mag)
    phase_scores = np.array(all_phase)
    coh_scores = np.array(all_coh)
    labels = np.array(all_labels) if all_labels else None

    return scores, mag_scores, phase_scores, coh_scores, labels


def compute_threshold(val_scores: np.ndarray,
                      percentile: float = THRESHOLD_PERCENTILE) -> float:
    """
    Compute anomaly threshold as the given percentile of validation scores.
    """
    threshold = np.percentile(val_scores, percentile)
    print(f"  Threshold (τ_{percentile/100:.2f}): {threshold:.6f}")
    return threshold


def classify(scores: np.ndarray, threshold: float) -> np.ndarray:
    """
    Classify windows as normal (0) or attack (1) based on threshold.
    """
    return (scores > threshold).astype(int)


if __name__ == "__main__":
    from dataset import get_dataloaders

    print("\n  Loading data and model ...")
    _, val_loader, attack_loader, _ = get_dataloaders()
    model = load_model()
    criterion = PhaseNetLoss()

    # Validation scores for threshold
    print("\n  Computing validation scores ...")
    val_scores, _, _, _, _ = compute_anomaly_scores(model, val_loader, criterion)
    threshold = compute_threshold(val_scores)

    # Attack scores
    print("\n  Computing attack scores ...")
    atk_scores, atk_mag, atk_phase, atk_coh, atk_labels = compute_anomaly_scores(
        model, attack_loader, criterion
    )

    predictions = classify(atk_scores, threshold)
    print(f"\n  Attack windows: {len(atk_scores)}")
    print(f"  Predicted attacks: {predictions.sum()}")
    print(f"  Score range: [{atk_scores.min():.4f}, {atk_scores.max():.4f}]")

    # Save scores
    np.savez(
        os.path.join(OUTPUT_DIR, "anomaly_scores.npz"),
        scores=atk_scores,
        mag_scores=atk_mag,
        phase_scores=atk_phase,
        coh_scores=atk_coh,
        labels=atk_labels,
        predictions=predictions,
        threshold=threshold,
    )
    print(f"  Scores saved to {OUTPUT_DIR}/anomaly_scores.npz")
