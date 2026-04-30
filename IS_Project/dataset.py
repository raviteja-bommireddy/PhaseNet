"""
dataset.py — PyTorch Dataset and DataLoader definitions for PhaseNet++.

Provides:
  • SWaTDataset      — map-style dataset that returns pre-computed STFT + PCI tensors
  • get_dataloaders  — builds train / val / attack DataLoaders
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config import (
    NORMAL_CSV, ATTACK_CSV, WINDOW_SIZE, STRIDE, N_FFT, FREQ_BINS,
    NUM_SENSORS, BATCH_SIZE, VAL_SPLIT, TEST_SPLIT, NUM_WORKERS, SEED, DEVICE,
)
from preprocessing.stft_pci import preprocess_window


# ──────────────────────────────────────────────────────────────
def _load_sensor_data(csv_path: str) -> tuple:
    """
    Load a SWaT CSV, extract numeric sensor columns, and return
    the sensor array and (optionally) labels.

    Returns:
        data:   (T, C) float32 numpy array
        labels: (T,)   numpy array of 0/1   (None if no label column)
    """
    print(f"  Loading {csv_path} ...")
    df = pd.read_csv(csv_path)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Identify label column
    label_col = None
    if "Normal/Attack" in df.columns:
        label_col = "Normal/Attack"

    # Sensor columns — everything except Timestamp and label
    exclude = {"Timestamp", "Normal/Attack"}
    sensor_cols = [c for c in df.columns if c not in exclude]

    # Convert to numeric (some SWaT columns have spaces)
    data = df[sensor_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)

    labels = None
    if label_col is not None:
        raw_labels = df[label_col].str.strip().values
        labels = np.where(raw_labels == "Attack", 1, 0).astype(np.int64)

    print(f"    shape={data.shape}, sensors={len(sensor_cols)}")
    return data, labels


# ──────────────────────────────────────────────────────────────
class SWaTDataset(Dataset):
    """
    Sliding-window dataset over pre-normalised sensor data.
    Each __getitem__ returns:
        stft_tensor:  (C, 2, F)   float32   — magnitude + phase
        pci_matrix:   (C, C)      float32   — phase coherence
    """

    def __init__(self, data: np.ndarray, window_size: int, stride: int,
                 n_fft: int, labels: np.ndarray = None):
        """
        Args:
            data:    (T, C) normalised sensor readings.
            window_size: W.
            stride:  sliding window stride.
            n_fft:   STFT parameter.
            labels:  optional (T,) array of 0/1.
        """
        self.data = data
        self.window_size = window_size
        self.stride = stride
        self.n_fft = n_fft
        self.labels = labels

        self.num_windows = (len(data) - window_size) // stride + 1
        print(f"    Initialized SWaTDataset for {self.num_windows} lazily tracked windows.")

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        
        # Slice lazily from the source continuous array
        window = self.data[start:end]  # (W, C)
        
        stft_t, pci_m = preprocess_window(window, n_fft=self.n_fft)
        
        stft = torch.from_numpy(stft_t)   # (C, 2, F)
        pci = torch.from_numpy(pci_m)     # (C, C)
        
        if self.labels is not None:
            # Check if any attack exists within this window
            label = int(self.labels[start:end].max())
            return stft, pci, label
            
        return stft, pci


# ──────────────────────────────────────────────────────────────
def get_dataloaders(max_train_windows: int = None):
    """
    Build train, val, and attack DataLoaders.

    Args:
        max_train_windows: if set, subsample normal data windows to this count
                           (useful for faster iteration on Kaggle).

    Returns:
        train_loader, val_loader, attack_loader, scaler
    """
    # 1. Load raw data
    normal_data, _ = _load_sensor_data(NORMAL_CSV)
    attack_data, attack_labels = _load_sensor_data(ATTACK_CSV)

    # 2. Fit scaler on normal data only
    print("  Fitting StandardScaler on normal data ...")
    scaler = StandardScaler()
    normal_data = scaler.fit_transform(normal_data)
    attack_data = scaler.transform(attack_data)

    # 3. Train/val/test split
    n = len(normal_data)
    val_size = int(n * VAL_SPLIT)
    test_size = int(n * TEST_SPLIT)  # Hold out TEST_SPLIT (15%) for the test set
    
    train_data = normal_data[:n - val_size - test_size]
    val_data = normal_data[n - val_size - test_size:n - test_size]
    test_normal_data = normal_data[n - test_size:]

    print(f"  Train samples: {len(train_data)}, Val samples: {len(val_data)}, Test Normal holdout: {len(test_normal_data)}")

    # Combine test_normal_data with attack_data to have positive and negative samples
    test_normal_labels = np.zeros(len(test_normal_data), dtype=np.int64)
    combined_test_data = np.vstack((test_normal_data, attack_data))
    combined_test_labels = np.concatenate((test_normal_labels, attack_labels))

    # Use larger stride for training to reduce preprocessing time on Kaggle
    train_stride = STRIDE
    if max_train_windows is not None:
        possible = (len(train_data) - WINDOW_SIZE) // 1 + 1
        if possible > max_train_windows:
            train_stride = max(1, (len(train_data) - WINDOW_SIZE) // max_train_windows)
            print(f"  Using train stride={train_stride} to cap at ~{max_train_windows} windows")

    # 4. Create datasets
    print("  Building training dataset ...")
    train_ds = SWaTDataset(train_data, WINDOW_SIZE, train_stride, N_FFT)
    print("  Building validation dataset ...")
    val_ds = SWaTDataset(val_data, WINDOW_SIZE, WINDOW_SIZE, N_FFT)  # non-overlapping
    print("  Building attack dataset ...")
    attack_ds = SWaTDataset(combined_test_data, WINDOW_SIZE, WINDOW_SIZE, N_FFT, labels=combined_test_labels)

    # 5. DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    attack_loader = DataLoader(attack_ds, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True)

    print(f"  DataLoaders ready — train batches={len(train_loader)}, "
          f"val batches={len(val_loader)}, attack batches={len(attack_loader)}")

    return train_loader, val_loader, attack_loader, scaler
