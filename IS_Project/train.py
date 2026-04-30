"""
train.py — Training loop with early stopping and checkpointing.

Trains PhaseNet++ exclusively on normal data. The model learns the manifold
of healthy system behavior; anomalies lie off this manifold and reconstruct
poorly, producing a high L_total.
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from config import (
    DEVICE, EPOCHS, LR, WEIGHT_DECAY, PATIENCE, GRAD_CLIP,
    CHECKPOINT_PATH, SEED, OUTPUT_DIR, BATCH_SIZE,
)
from model.phasenet import PhaseNetPP
from model.losses import PhaseNetLoss
from dataset import get_dataloaders


def set_seed(seed: int = SEED):
    """Reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, scaler, grad_clip):
    """Run a single training epoch."""
    model.train()
    total_loss = 0.0
    total_mag = 0.0
    total_phase = 0.0
    total_coh = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for batch in pbar:
        stft, pci = batch[0].to(DEVICE), batch[1].to(DEVICE)

        # Extract targets: magnitude = stft[:, :, 0, :], phase = stft[:, :, 1, :]
        mag_target = stft[:, :, 0, :]   # (B, C, F)
        phase_target = stft[:, :, 1, :]  # (B, C, F)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            mag_hat, phase_hat = model(stft, pci)
            loss, (ml, pl, cl) = criterion(mag_hat, phase_hat,
                                           mag_target, phase_target, pci)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_mag += ml
        total_phase += pl
        total_coh += cl
        n_batches += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return (total_loss / n_batches, total_mag / n_batches,
            total_phase / n_batches, total_coh / n_batches)


@torch.no_grad()
def validate(model, loader, criterion):
    """Run validation."""
    model.eval()
    total_loss = 0.0
    total_mag = 0.0
    total_phase = 0.0
    total_coh = 0.0
    n_batches = 0

    for batch in loader:
        stft, pci = batch[0].to(DEVICE), batch[1].to(DEVICE)
        mag_target = stft[:, :, 0, :]
        phase_target = stft[:, :, 1, :]

        with autocast():
            mag_hat, phase_hat = model(stft, pci)
            loss, (ml, pl, cl) = criterion(mag_hat, phase_hat,
                                           mag_target, phase_target, pci)

        total_loss += loss.item()
        total_mag += ml
        total_phase += pl
        total_coh += cl
        n_batches += 1

    return (total_loss / n_batches, total_mag / n_batches,
            total_phase / n_batches, total_coh / n_batches)


def train(max_train_windows: int = None):
    """
    Full training pipeline.

    Args:
        max_train_windows: cap on training windows for faster Kaggle runs.
    """
    set_seed()
    print(f"\n{'='*60}")
    print(f"  PhaseNet++ Training")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}\n")

    # Data
    train_loader, val_loader, attack_loader, scaler = get_dataloaders(
        max_train_windows=max_train_windows
    )

    # Model
    model = PhaseNetPP().to(DEVICE)
    print(f"\n  Model parameters: {model.count_parameters():,}")

    # Loss, Optimizer, Scheduler
    criterion = PhaseNetLoss().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    amp_scaler = GradScaler()

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}

    print(f"\n  Starting training for up to {EPOCHS} epochs ...\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_mag, train_phase, train_coh = train_one_epoch(
            model, train_loader, criterion, optimizer, amp_scaler, GRAD_CLIP
        )

        val_loss, val_mag, val_phase, val_coh = validate(model, val_loader, criterion)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(lr)

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, CHECKPOINT_PATH)
            improved = " ✓ saved"
        else:
            patience_counter += 1

        print(f"  Epoch {epoch:3d}/{EPOCHS} │ "
              f"Train {train_loss:.4f} (M={train_mag:.4f} P={train_phase:.4f} C={train_coh:.4f}) │ "
              f"Val {val_loss:.4f} │ LR {lr:.2e} │ {elapsed:.1f}s{improved}")

        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

    print(f"\n  Best validation loss: {best_val_loss:.4f}")
    print(f"  Checkpoint saved to: {CHECKPOINT_PATH}")

    # Save training history
    import json
    hist_path = os.path.join(OUTPUT_DIR, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f)
    print(f"  Training history saved to: {hist_path}")

    # Plot training curves
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(history["train_loss"], label="Train Loss", linewidth=2)
        ax1.plot(history["val_loss"], label="Val Loss", linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(history["lr"], label="Learning Rate", color="green", linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("Learning Rate Schedule")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, "training_curves.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Training curves saved to: {plot_path}")
    except Exception as e:
        print(f"  Warning: Could not plot training curves: {e}")

    return model, attack_loader, scaler


if __name__ == "__main__":
    # For Kaggle: use max_train_windows to limit preprocessing time
    # Set to None for full training, or e.g. 5000 for a quick test
    max_windows = None
    if len(sys.argv) > 1:
        max_windows = int(sys.argv[1])
    train(max_train_windows=max_windows)
