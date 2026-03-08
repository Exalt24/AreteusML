"""Classical ML baselines (TF-IDF + classifier) with MLflow tracking.

Run with:
    python -m ml.training.train_baseline
"""

from __future__ import annotations

from pathlib import Path

import pickle

import mlflow
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ml.training.evaluate import (
    compute_metrics,
    log_metrics_to_mlflow,
    per_class_report,
    plot_confusion_matrix,
)
from ml.utils.reproducibility import SEED, set_seed

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "baseline"

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
TFIDF_PARAMS: dict = {
    "max_features": 10_000,
    "ngram_range": (1, 2),
    "sublinear_tf": True,
}

MODELS: dict[str, object] = {
    "TF-IDF + LogReg": LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=SEED),
    "TF-IDF + SVM": LinearSVC(class_weight="balanced", random_state=SEED, max_iter=2000),
    "TF-IDF + RandomForest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=-1
    ),
}


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test parquet files from the processed data directory."""
    train_path = DATA_DIR / "train.parquet"
    test_path = DATA_DIR / "test.parquet"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Processed data not found in {DATA_DIR}. Run the data pipeline first.")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    logger.info(f"Loaded train ({len(train_df)}) and test ({len(test_df)}) splits")
    return train_df, test_df


def train_and_evaluate(
    name: str,
    classifier: object,
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
    label_names: list[str],
) -> dict:
    """Train a single TF-IDF pipeline, evaluate, and log to MLflow."""
    logger.info(f"Training {name}...")

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
            ("clf", classifier),
        ]
    )

    with mlflow.start_run(run_name=name, nested=True):
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        metrics = compute_metrics(y_test, y_pred)
        log_metrics_to_mlflow(metrics)

        # Log parameters
        mlflow.log_params(
            {
                "model": name,
                "tfidf_max_features": TFIDF_PARAMS["max_features"],
                "tfidf_ngram_range": str(TFIDF_PARAMS["ngram_range"]),
                "tfidf_sublinear_tf": TFIDF_PARAMS["sublinear_tf"],
                "seed": SEED,
            }
        )

        # Confusion matrix artifact
        cm_path = ARTIFACTS_DIR / f"{name.replace(' ', '_')}_confusion_matrix.png"
        plot_confusion_matrix(y_test, y_pred, label_names, save_path=cm_path)
        mlflow.log_artifact(str(cm_path))

        # Per-class report artifact
        report_df = per_class_report(y_test, y_pred, label_names)
        report_path = ARTIFACTS_DIR / f"{name.replace(' ', '_')}_per_class.csv"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(report_path)
        mlflow.log_artifact(str(report_path))

        logger.success(f"{name} -> Accuracy: {metrics['accuracy']:.4f}  F1 (macro): {metrics['f1_macro']:.4f}")

    return {"model": name, "pipeline": pipe, **metrics}


def print_comparison_table(results: list[dict]) -> None:
    """Print a Rich table comparing all baseline models."""
    console = Console()
    table = Table(title="Baseline Model Comparison", show_lines=True)

    table.add_column("Model", style="bold cyan")
    table.add_column("Accuracy", justify="right")
    table.add_column("F1 (macro)", justify="right")
    table.add_column("F1 (micro)", justify="right")
    table.add_column("F1 (weighted)", justify="right")
    table.add_column("P (macro)", justify="right")
    table.add_column("R (macro)", justify="right")

    # Sort by macro F1 descending
    results_sorted = sorted(results, key=lambda r: r["f1_macro"], reverse=True)

    for r in results_sorted:
        table.add_row(
            r["model"],
            f"{r['accuracy']:.4f}",
            f"{r['f1_macro']:.4f}",
            f"{r['f1_micro']:.4f}",
            f"{r['f1_weighted']:.4f}",
            f"{r['precision_macro']:.4f}",
            f"{r['recall_macro']:.4f}",
        )

    console.print(table)

    best = results_sorted[0]
    console.print(f"\n[bold green]Best model:[/bold green] {best['model']} (F1 macro = {best['f1_macro']:.4f})")


def main() -> None:
    set_seed(SEED)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_splits()

    X_train, y_train = train_df["text"], train_df["label"]
    X_test, y_test = test_df["text"], test_df["label"]
    label_names = [str(i) for i in sorted(y_train.unique().tolist())]

    logger.info(f"Classes: {len(label_names)} | Train: {len(X_train)} | Test: {len(X_test)}")

    mlflow.set_experiment("banking77-baselines")

    results: list[dict] = []

    with mlflow.start_run(run_name="baseline-sweep"):
        for name, clf in MODELS.items():
            metrics = train_and_evaluate(name, clf, X_train, y_train, X_test, y_test, label_names)
            results.append(metrics)

        # Log the best model's metrics at the parent run level
        best = max(results, key=lambda r: r["f1_macro"])
        mlflow.log_metric("best_f1_macro", best["f1_macro"])
        mlflow.set_tag("best_model", best["model"])

    # Save best pipeline to disk for SHAP analysis
    pipeline_path = ARTIFACTS_DIR / "best_pipeline.pkl"
    with open(pipeline_path, "wb") as f:
        pickle.dump(best["pipeline"], f)
    logger.info(f"Saved best pipeline to {pipeline_path}")

    print_comparison_table(results)


if __name__ == "__main__":
    main()
