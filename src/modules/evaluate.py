"""Evaluation utilities for the hybrid log anomaly detector.

Computes metrics, generates plots, and writes results consumed by CML.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> dict[str, float]:
    """Compute core anomaly-detection metrics."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auroc = roc_auc_score(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    auprc = auc(recall, precision)

    y_pred = (y_scores > 0.5).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "auroc": round(float(auroc), 4),
        "auprc": round(float(auprc), 4),
        "f1": round(float(f1), 4),
    }


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, out_path: str | Path = "plots/roc.png") -> None:
    """Generate and save an ROC curve plot."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_scores):.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Anomaly Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_loss_curve(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    out_path: str | Path = "plots/loss.png",
) -> None:
    """Generate and save a training loss curve."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(train_losses, label="Train")
    if val_losses:
        plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_metrics(metrics: dict[str, Any], path: str | Path = "metrics.json") -> None:
    """Write metrics dict to JSON (consumed by DVC & CML)."""
    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Metrics saved → {path}")
