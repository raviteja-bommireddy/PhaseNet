"""
config.py — All hyperparameters and configuration for PhaseNet++.
Centralised so every module reads from one source of truth.
"""

import os
import torch

# ──────────────────────────────────────────────
# Paths  (auto-detect Kaggle vs local)
# ──────────────────────────────────────────────
IS_KAGGLE = os.path.exists("/kaggle")

if IS_KAGGLE:
    # Kaggle dataset path
    _DATA_ROOT = "/kaggle/input/datasets/pakalalohith/is-project/IS_Project/dataset"
    OUTPUT_DIR = "/kaggle/working/output"
else:
    _DATA_ROOT = os.path.join(os.path.dirname(__file__), "dataset")
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

NORMAL_CSV = os.path.join(_DATA_ROOT, "normal.csv")
ATTACK_CSV = os.path.join(_DATA_ROOT, "attack.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "model1.pt")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# Dataset / Preprocessing
# ──────────────────────────────────────────────
NUM_SENSORS = 51          # C — SWaT dataset columns (excluding Timestamp & label)
WINDOW_SIZE = 60         # W — sliding window length
STRIDE = 5               # sliding window stride (Updated from 1 for 1M timestamps)
N_FFT = 128              # STFT window
FREQ_BINS = N_FFT // 2 + 1   # F = 65

# ──────────────────────────────────────────────
# Model dimensions
# ──────────────────────────────────────────────
EMBED_DIM = 128           # D — node embedding dimension from CNN
D_MODEL = 256             # Transformer hidden dimension
N_HEADS = 8               # attention heads (GAT + Transformer)
NUM_TRANSFORMER_LAYERS = 4
FFN_EXPANSION = 4
DROPOUT = 0.20

# ──────────────────────────────────────────────
# Loss weights
# ──────────────────────────────────────────────
ALPHA = 0.5   # magnitude MSE weight
BETA  = 3.0   # phase circular loss weight
GAMMA = 1.5  # coherence MSE weight

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
BATCH_SIZE = 32          # optimal for convergence quality and 30GB VRAM
EPOCHS = 80               # model converges ~60-80 epochs with early stopping
LR = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 15             # early stopping patience
GRAD_CLIP = 1.0
VAL_SPLIT = 0.10          # fraction of normal data for validation
TEST_SPLIT = 0.15         # fraction of normal data held out for testing

# ──────────────────────────────────────────────
# Anomaly detection
# ──────────────────────────────────────────────
THRESHOLD_PERCENTILE = 99 # τ₀.₉₉

# ──────────────────────────────────────────────
# Misc
# ──────────────────────────────────────────────
SEED = 42
NUM_WORKERS = 4 if IS_KAGGLE else 0
