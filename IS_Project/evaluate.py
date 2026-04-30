"""
evaluate.py — Precision / Recall / F1 against attack labels.

Produces:
  1. Classification metrics (precision, recall, F1, accuracy)
  2. Confusion matrix visualization
  3. Score distribution plots
  4. ROC / Precision-Recall curves
  5. Per-component anomaly fingerprint analysis
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)

from config import DEVICE, CHECKPOINT_PATH, OUTPUT_DIR, RESULTS_DIR, THRESHOLD_PERCENTILE
from model.phasenet import PhaseNetPP
from model.losses import PhaseNetLoss
from inference import load_model, compute_anomaly_scores, compute_threshold, classify
from dataset import get_dataloaders


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title("Confusion Matrix", fontsize=15, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Attack"], fontsize=12)
    ax.set_yticklabels(["Normal", "Attack"], fontsize=12)

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    fontsize=16, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    Saved: {save_path}")


def plot_score_distributions(normal_scores, attack_scores, threshold, save_path):
    """Plot score distributions for normal vs attack windows."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(normal_scores, bins=100, alpha=0.6, label="Normal", color="#2196F3",
            density=True, edgecolor="none")
    ax.hist(attack_scores, bins=100, alpha=0.6, label="Attack", color="#F44336",
            density=True, edgecolor="none")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=2,
               label=f"Threshold τ₀.₉₆ = {threshold:.4f}")
    ax.set_xlabel("Anomaly Score (L_total)", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title("Anomaly Score Distribution", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    Saved: {save_path}")


def plot_roc_pr_curves(y_true, scores, save_path):
    """Plot ROC and Precision-Recall curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, linewidth=2, color="#2196F3", label=f"ROC (AUC = {roc_auc:.4f})")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax1.set_xlabel("False Positive Rate", fontsize=13)
    ax1.set_ylabel("True Positive Rate", fontsize=13)
    ax1.set_title("ROC Curve", fontsize=15, fontweight="bold")
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    ax2.plot(recall, precision, linewidth=2, color="#F44336", label=f"PR (AP = {ap:.4f})")
    ax2.set_xlabel("Recall", fontsize=13)
    ax2.set_ylabel("Precision", fontsize=13)
    ax2.set_title("Precision-Recall Curve", fontsize=15, fontweight="bold")
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    Saved: {save_path}")


def plot_component_analysis(mag_scores, phase_scores, coh_scores, labels, save_path):
    """Plot per-component score analysis for attack fingerprinting."""
    normal_mask = labels == 0
    attack_mask = labels == 1

    components = ["Magnitude", "Phase", "Coherence"]
    normal_means = [
        mag_scores[normal_mask].mean(),
        phase_scores[normal_mask].mean(),
        coh_scores[normal_mask].mean(),
    ]
    attack_means = [
        mag_scores[attack_mask].mean(),
        phase_scores[attack_mask].mean(),
        coh_scores[attack_mask].mean(),
    ]

    x = np.arange(len(components))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, normal_means, width, label="Normal", color="#4CAF50", alpha=0.8)
    bars2 = ax.bar(x + width/2, attack_means, width, label="Attack", color="#F44336", alpha=0.8)

    ax.set_xlabel("Loss Component", fontsize=13)
    ax.set_ylabel("Mean Score", fontsize=13)
    ax.set_title("Anomaly Fingerprint — Per-Component Analysis", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(components, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h, f"{h:.4f}",
                    ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    Saved: {save_path}")


def evaluate(max_train_windows: int = None):
    """
    Full evaluation pipeline.
    """
    print(f"\n{'='*60}")
    print(f"  PhaseNet++ Evaluation")
    print(f"{'='*60}\n")

    # Load data
    train_loader, val_loader, attack_loader, scaler = get_dataloaders(
        max_train_windows=max_train_windows
    )

    # Load model
    model = load_model()
    criterion = PhaseNetLoss()

    # 1. Validation scores → threshold
    print("\n  Computing validation anomaly scores ...")
    val_scores, _, _, _, _ = compute_anomaly_scores(model, val_loader, criterion)
    threshold = compute_threshold(val_scores)

    # 2. Attack scores
    print("\n  Computing attack anomaly scores ...")
    atk_scores, atk_mag, atk_phase, atk_coh, atk_labels = compute_anomaly_scores(
        model, attack_loader, criterion
    )

    if atk_labels is None:
        print("  ERROR: Attack dataset has no labels. Cannot evaluate.")
        return

    # 3. Classification
    predictions = classify(atk_scores, threshold)

    # 4. Metrics
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(classification_report(atk_labels, predictions,
                                target_names=["Normal", "Attack"], digits=4))

    prec = precision_score(atk_labels, predictions)
    rec = recall_score(atk_labels, predictions)
    f1 = f1_score(atk_labels, predictions)
    acc = accuracy_score(atk_labels, predictions)

    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Threshold : {threshold:.6f}")

    # ROC AUC
    fpr, tpr, _ = roc_curve(atk_labels, atk_scores)
    roc_auc = auc(fpr, tpr)
    ap = average_precision_score(atk_labels, atk_scores)
    print(f"  ROC AUC   : {roc_auc:.4f}")
    print(f"  Avg Prec  : {ap:.4f}")

    # 5. Visualizations
    print(f"\n  Generating visualizations ...")

    # Separate scores by label for distribution plots
    normal_in_attack = atk_scores[atk_labels == 0]
    attack_only = atk_scores[atk_labels == 1]

    plot_confusion_matrix(
        atk_labels, predictions,
        os.path.join(RESULTS_DIR, "confusion_matrix.png")
    )

    plot_score_distributions(
        normal_in_attack, attack_only, threshold,
        os.path.join(RESULTS_DIR, "score_distributions.png")
    )

    plot_roc_pr_curves(
        atk_labels, atk_scores,
        os.path.join(RESULTS_DIR, "roc_pr_curves.png")
    )

    plot_component_analysis(
        atk_mag, atk_phase, atk_coh, atk_labels,
        os.path.join(RESULTS_DIR, "component_analysis.png")
    )

    # Save metrics summary
    metrics = {
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "accuracy": float(acc),
        "roc_auc": float(roc_auc),
        "avg_precision": float(ap),
        "threshold": float(threshold),
        "num_attack_windows": int(atk_labels.sum()),
        "num_normal_windows": int((atk_labels == 0).sum()),
        "num_predicted_attacks": int(predictions.sum()),
    }

    import json
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved to: {metrics_path}")
    print(f"  All results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    max_windows = None
    if len(sys.argv) > 1:
        max_windows = int(sys.argv[1])
    evaluate(max_train_windows=max_windows)
