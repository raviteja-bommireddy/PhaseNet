"""
preprocessing/data_explorer.py — Dataset statistics and visualization utilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from config import NORMAL_CSV, ATTACK_CSV, RESULTS_DIR


def load_and_summarise(csv_path: str, name: str = "dataset") -> pd.DataFrame:
    """Load a CSV and print summary statistics."""
    print(f"\n{'='*60}")
    print(f"  Loading {name}: {csv_path}")
    print(f"{'='*60}")
    df = pd.read_csv(csv_path)
    print(f"  Shape           : {df.shape}")
    print(f"  Columns         : {list(df.columns[:5])} ... {list(df.columns[-3:])}")
    print(f"  Dtypes (unique) : {df.dtypes.value_counts().to_dict()}")

    # Identify numeric sensor columns (exclude Timestamp and label)
    sensor_cols = [c for c in df.columns if c.strip() not in ("Timestamp", "Normal/Attack")]
    numeric = df[sensor_cols].apply(pd.to_numeric, errors="coerce")
    print(f"  Sensor columns  : {len(sensor_cols)}")
    print(f"  NaN count       : {numeric.isna().sum().sum()}")
    print(f"  Value range     : [{numeric.min().min():.4f}, {numeric.max().max():.4f}]")
    return df


def plot_sensor_distributions(df: pd.DataFrame, tag: str = "normal"):
    """Plot histograms of a few representative sensors."""
    sensor_cols = [c for c in df.columns if c.strip() not in ("Timestamp", "Normal/Attack")]
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for idx, ax in enumerate(axes.flat):
        if idx < len(sensor_cols):
            col = sensor_cols[idx]
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            ax.hist(vals, bins=50, alpha=0.7, edgecolor="black")
            ax.set_title(col.strip(), fontsize=9)
        else:
            ax.axis("off")
    plt.suptitle(f"Sensor Distributions ({tag})", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"sensor_distributions_{tag}.png"), dpi=150)
    plt.close()
    print(f"  Saved sensor_distributions_{tag}.png")


if __name__ == "__main__":
    df_normal = load_and_summarise(NORMAL_CSV, "Normal")
    plot_sensor_distributions(df_normal, "normal")

    df_attack = load_and_summarise(ATTACK_CSV, "Attack")
    plot_sensor_distributions(df_attack, "attack")

    # Attack label distribution
    if "Normal/Attack" in df_attack.columns:
        col = "Normal/Attack"
    else:
        col = df_attack.columns[-1]
    print(f"\n  Attack label distribution:\n{df_attack[col].value_counts()}")
