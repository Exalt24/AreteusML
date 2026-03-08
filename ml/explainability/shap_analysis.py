"""SHAP explainability analysis for the best classical baseline model.

Generates summary plots, force plots, and top features per class
using shap.LinearExplainer on the TF-IDF + LogReg pipeline.

Run with:
    python -m ml.explainability.shap_analysis
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from loguru import logger
from scipy.sparse import issparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "ml" / "explainability" / "outputs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "baseline"
PIPELINE_PATH = ARTIFACTS_DIR / "best_pipeline.pkl"

N_DISPLAY_FEATURES = 20
N_FORCE_SAMPLES = 5


def load_pipeline():
    """Load TF-IDF + LogReg pipeline from pickle or MLflow."""
    if PIPELINE_PATH.exists():
        logger.info(f"Loading pipeline from {PIPELINE_PATH}")
        with open(PIPELINE_PATH, "rb") as f:
            return pickle.load(f)

    logger.info("Pickle not found, attempting MLflow model load...")
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Banking77-Baselines")
    if experiment is None:
        raise FileNotFoundError(
            f"No pipeline found at {PIPELINE_PATH} and no MLflow experiment "
            "'Banking77-Baselines' exists. Train baselines first."
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'TF-IDF + LogReg'",
        order_by=["metrics.f1_macro DESC"],
        max_results=1,
    )
    if not runs:
        raise FileNotFoundError("No TF-IDF + LogReg run found in MLflow")

    model_uri = f"runs:/{runs[0].info.run_id}/model"
    return mlflow.sklearn.load_model(model_uri)


def load_test_data() -> tuple[pd.Series, pd.Series, list[str]]:
    """Load test split and return texts, labels, and label names."""
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")
    label_names = sorted(test_df["label"].unique().tolist()) if test_df["label"].dtype == object else None

    if label_names is None:
        train_df = pd.read_parquet(DATA_DIR / "train.parquet")
        all_labels = (
            sorted(train_df["label_text"].unique().tolist())
            if "label_text" in train_df.columns
            else [str(i) for i in range(test_df["label"].nunique())]
        )
        label_names = all_labels

    return test_df["text"], test_df["label"], label_names


def run_shap_analysis(pipeline, X_test: pd.Series, label_names: list[str]) -> None:
    """Run SHAP analysis and save plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    feature_names = np.array(tfidf.get_feature_names_out())

    # Subsample for memory efficiency (77 classes x 10k features is huge)
    n_shap_samples = min(200, len(X_test))
    X_sample = X_test.sample(n=n_shap_samples, random_state=42).reset_index(drop=True)
    logger.info(f"Using {n_shap_samples} samples for SHAP analysis (memory constraint)")

    X_tfidf = tfidf.transform(X_sample)
    X_dense = X_tfidf.toarray() if issparse(X_tfidf) else np.asarray(X_tfidf)

    logger.info("Computing SHAP values with LinearExplainer...")
    explainer = shap.LinearExplainer(clf, X_dense, feature_names=feature_names)
    shap_values = explainer.shap_values(X_dense)

    # --- Summary plot (global feature importance) ---
    logger.info("Generating summary plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_dense,
        feature_names=feature_names,
        max_display=N_DISPLAY_FEATURES,
        show=False,
        plot_type="bar",
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.success("Saved shap_summary.png")

    # --- Force plots skipped for 77-class model (SHAP force_plot doesn't handle well) ---
    logger.info("Skipping force plots (not practical for 77-class classification)")

    # --- Top features per class ---
    logger.info("Computing top features per class...")
    if isinstance(shap_values, list):
        n_classes = len(shap_values)
    elif shap_values.ndim == 3:
        n_classes = shap_values.shape[-1]
    else:
        n_classes = 1

    top_features = {}
    for c in range(min(n_classes, len(label_names))):
        if isinstance(shap_values, list):
            class_shap = np.abs(shap_values[c]).mean(axis=0)
        elif shap_values.ndim == 3:
            class_shap = np.abs(shap_values[:, :, c]).mean(axis=0)
        else:
            class_shap = np.abs(shap_values).mean(axis=0)

        top_idx = np.argsort(class_shap)[-N_DISPLAY_FEATURES:][::-1]
        top_features[label_names[c]] = [
            {"feature": str(feature_names[j]), "importance": float(class_shap[j])} for j in top_idx
        ]

    import json

    with open(OUTPUT_DIR / "top_features_per_class.json", "w") as f:
        json.dump(top_features, f, indent=2)
    logger.success("Saved top_features_per_class.json")


def main() -> None:
    """Run SHAP explainability analysis."""
    pipeline = load_pipeline()
    X_test, y_test, label_names = load_test_data()

    logger.info(f"Test set: {len(X_test)} samples, {len(label_names)} classes")
    run_shap_analysis(pipeline, X_test, label_names)
    logger.success(f"All SHAP outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
