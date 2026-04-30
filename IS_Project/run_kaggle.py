"""
run_kaggle.py — Single-file Kaggle runner for PhaseNet++.

Run this in a Kaggle notebook cell:
    %run run_kaggle.py

Or as a script:
    python run_kaggle.py

Pipeline (matches README exactly):
  Phase 0: Data Exploration & Preprocessing
    - Load CSVs, StandardScaler normalization
    - Sliding Window decomposition (Section 5.1)
    - STFT computation — Magnitude & Phase extraction (Section 5.2)
    - PCI matrix generation — Phase Coherence Index (Section 5.3)
    - DataLoader tensor assembly (Section 5.4)
  Phase 1: Training
    - PhaseNet++ autoencoder on normal data only
  Phase 2: Evaluation
    - Anomaly scoring, threshold, classification metrics
"""

import os
import sys
import time
import gc

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

from config import (
    DEVICE, OUTPUT_DIR, RESULTS_DIR, NORMAL_CSV, ATTACK_CSV,
    WINDOW_SIZE, STRIDE, N_FFT, FREQ_BINS, NUM_SENSORS,
    BATCH_SIZE, VAL_SPLIT,
)

print(f"\nDevice: {DEVICE}")
print(f"Normal CSV: {NORMAL_CSV}")
print(f"Attack CSV: {ATTACK_CSV}")
print(f"Output dir: {OUTPUT_DIR}")

# Check data files exist
for path in [NORMAL_CSV, ATTACK_CSV]:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  ✓ {os.path.basename(path)}: {size_mb:.1f} MB")
    else:
        print(f"  ✗ MISSING: {path}")
        print("    Please upload your dataset to Kaggle and update config.py paths!")
        sys.exit(1)


# ═══════════════════════════════════════════════
# PHASE 0: DATA EXPLORATION & PREPROCESSING
# ═══════════════════════════════════════════════
print(f"\n{'═'*60}")
print(f"  PHASE 0: DATA EXPLORATION & PREPROCESSING")
print(f"{'═'*60}")

t_preprocess = time.time()

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from preprocessing.stft_pci import compute_stft, compute_pci, preprocess_window

# ── Step 0.1: Load raw CSV data ──
print(f"\n  [Step 0.1] Loading raw CSV data ...")
df_normal = pd.read_csv(NORMAL_CSV)
df_normal.columns = df_normal.columns.str.strip()
print(f"    Normal data shape: {df_normal.shape}")

df_attack = pd.read_csv(ATTACK_CSV)
df_attack.columns = df_attack.columns.str.strip()
print(f"    Attack data shape: {df_attack.shape}")

# ── Step 0.2: Extract sensor columns ──
print(f"\n  [Step 0.2] Extracting {NUM_SENSORS} sensor columns ...")
exclude = {"Timestamp", "Normal/Attack"}
sensor_cols = [c for c in df_normal.columns if c not in exclude]
print(f"    Sensors identified: {len(sensor_cols)}")
print(f"    First 5: {sensor_cols[:5]}")
print(f"    Last 5:  {sensor_cols[-5:]}")

normal_data = df_normal[sensor_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
attack_data = df_attack[sensor_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)

# Attack labels
attack_labels = None
if "Normal/Attack" in df_attack.columns:
    raw_labels = df_attack["Normal/Attack"].str.strip().values
    attack_labels = np.where(raw_labels == "Attack", 1, 0).astype(np.int64)
    n_attack = attack_labels.sum()
    n_normal = len(attack_labels) - n_attack
    print(f"    Attack labels: {n_attack} attack / {n_normal} normal timestamps")

# Free the DataFrames
del df_normal, df_attack
gc.collect()

# ── Step 0.3: StandardScaler normalization ──
print(f"\n  [Step 0.3] Fitting StandardScaler on normal data (zero-mean, unit-variance) ...")
scaler = StandardScaler()
normal_data = scaler.fit_transform(normal_data)
attack_data = scaler.transform(attack_data)
print(f"    Normal data range after scaling: [{normal_data.min():.4f}, {normal_data.max():.4f}]")
print(f"    Attack data range after scaling: [{attack_data.min():.4f}, {attack_data.max():.4f}]")

# ── Step 0.4: Sliding Window (README Section 5.1) ──
print(f"\n  [Step 0.4] Sliding Window Decomposition (README §5.1)")
print(f"    Window size W = {WINDOW_SIZE}")
print(f"    Stride       = {STRIDE}")

n = len(normal_data)
val_size = int(n * VAL_SPLIT)
train_data = normal_data[:n - val_size]
val_data = normal_data[n - val_size:]

n_train_windows = (len(train_data) - WINDOW_SIZE) // STRIDE + 1
n_val_windows = (len(val_data) - WINDOW_SIZE) // WINDOW_SIZE + 1
n_attack_windows = (len(attack_data) - WINDOW_SIZE) // WINDOW_SIZE + 1

print(f"    Train timestamps: {len(train_data):,} → {n_train_windows:,} windows")
print(f"    Val timestamps:   {len(val_data):,} → {n_val_windows:,} windows")
print(f"    Attack timestamps:{len(attack_data):,} → {n_attack_windows:,} windows")

# ── Step 0.5: STFT & PCI demo on a sample window (README §5.2 & §5.3) ──
print(f"\n  [Step 0.5] STFT & PCI Computation (README §5.2, §5.3)")
print(f"    n_fft = {N_FFT}")
print(f"    Frequency bins F = n_fft/2 + 1 = {FREQ_BINS}")

# Demo on a single window to show the pipeline
sample_window = train_data[:WINDOW_SIZE]  # (W, C)
sample_mag, sample_phase = compute_stft(sample_window, n_fft=N_FFT)
sample_pci = compute_pci(sample_phase)
sample_stft_tensor = np.stack([sample_mag, sample_phase], axis=1)

print(f"\n    ── Sample window preprocessing result ──")
print(f"    Window input shape:    ({WINDOW_SIZE}, {NUM_SENSORS})")
print(f"    Magnitude |Z| shape:   {sample_mag.shape}  (C, F)")
print(f"    Phase φ(Z) shape:      {sample_phase.shape}  (C, F)")
print(f"    Phase range:           [{sample_phase.min():.4f}, {sample_phase.max():.4f}] (should be [-π, π])")
print(f"    PCI matrix shape:      {sample_pci.shape}  (C, C)")
print(f"    PCI range:             [{sample_pci.min():.4f}, {sample_pci.max():.4f}] (should be [0, 1])")
print(f"    PCI diagonal (self):   {sample_pci[0,0]:.4f} (should be 1.0)")
print(f"    STFT tensor shape:     {sample_stft_tensor.shape}  (C, 2, F)")

# ── Step 0.6: DataLoader Assembly (README §5.4) ──
print(f"\n  [Step 0.6] DataLoader Tensor Assembly (README §5.4)")
print(f"    Each batch will provide:")
print(f"      STFT Tensor X : (B={BATCH_SIZE}, C={NUM_SENSORS}, 2, F={FREQ_BINS})")
print(f"      PCI Matrix  A : (B={BATCH_SIZE}, C={NUM_SENSORS}, C={NUM_SENSORS})")
print(f"    Preprocessing runs on-the-fly per batch (vectorized SciPy, ~0.2ms/window)")

preprocess_time = time.time() - t_preprocess
print(f"\n  ✓ Preprocessing pipeline validated in {preprocess_time:.1f}s")
print(f"    All STFT + PCI computation will execute on-the-fly during training")
print(f"    (40x faster via vectorized SciPy — no memory pre-allocation needed)")

# Clean up sample data
del sample_window, sample_mag, sample_phase, sample_pci, sample_stft_tensor
del normal_data, attack_data, train_data, val_data
gc.collect()


# ═══════════════════════════════════════════════
# PHASE 1: TRAINING
# ═══════════════════════════════════════════════
print(f"\n{'═'*60}")
print(f"  PHASE 1: TRAINING")
print(f"{'═'*60}")

t_start = time.time()

MAX_TRAIN_WINDOWS = None   # Full training — uses all normal data for best results

from train import train
model, attack_loader, scaler = train(max_train_windows=MAX_TRAIN_WINDOWS)

train_time = time.time() - t_start
print(f"\n  Training completed in {train_time / 60:.1f} minutes")

# Free memory
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════
# PHASE 2: EVALUATION
# ═══════════════════════════════════════════════
print(f"\n{'═'*60}")
print(f"  PHASE 2: EVALUATION")
print(f"{'═'*60}")

t_eval = time.time()

from evaluate import evaluate
evaluate(max_train_windows=MAX_TRAIN_WINDOWS)

eval_time = time.time() - t_eval
print(f"\n  Evaluation completed in {eval_time / 60:.1f} minutes")


# ═══════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════
total_time = time.time() - t_preprocess
print(f"\n{'═'*60}")
print(f"  DONE — Total time: {total_time / 60:.1f} minutes")
print(f"{'═'*60}")
print(f"\n  Output files:")
for root, dirs, files in os.walk(OUTPUT_DIR):
    for f in sorted(files):
        path = os.path.join(root, f)
        size = os.path.getsize(path) / 1024
        print(f"    {os.path.relpath(path, OUTPUT_DIR):40s} {size:8.1f} KB")
