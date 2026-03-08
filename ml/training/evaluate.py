"""Evaluation utilities for model performance analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict:
    """Compute classification metrics across averaging strategies.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    y_prob : array-like, optional
        Predicted probabilities (reserved for future use, e.g. log-loss).

    Returns
    -------
    dict
        Dictionary with accuracy, precision, recall, and F1 for
        macro / micro / weighted averaging.
    """
    metrics: dict = {"accuracy": accuracy_score(y_true, y_pred)}

    for avg in ("macro", "micro", "weighted"):
        metrics[f"precision_{avg}"] = precision_score(y_true, y_pred, average=avg, zero_division=0)
        metrics[f"recall_{avg}"] = recall_score(y_true, y_pred, average=avg, zero_division=0)
        metrics[f"f1_{avg}"] = f1_score(y_true, y_pred, average=avg, zero_division=0)

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot and optionally save a confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    labels : list[str]
        Ordered label names.
    save_path : str or Path, optional
        If provided, figure is saved to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes = len(labels)

    fig_size = max(10, n_classes * 0.25)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", fontsize=5)
    plt.setp(ax.get_yticklabels(), fontsize=5)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def per_class_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> pd.DataFrame:
    """Build a per-class metrics DataFrame.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    label_names : list[str]
        Ordered label names.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by class name with precision, recall, f1-score,
        and support columns.
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    df = pd.DataFrame(report).T
    # Keep only per-class rows (drop macro avg, weighted avg, accuracy)
    df = df.loc[label_names]
    return df


def log_metrics_to_mlflow(metrics: dict) -> None:
    """Log every key-value pair in *metrics* to the active MLflow run.

    Parameters
    ----------
    metrics : dict
        Flat dictionary of metric_name -> numeric value.
    """
    for key, value in metrics.items():
        mlflow.log_metric(key, value)
