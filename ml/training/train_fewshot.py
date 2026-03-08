"""SetFit few-shot experiments on Banking77 with learning curve analysis.

Run with:
    python -m ml.training.train_fewshot
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from setfit import SetFitModel
from setfit import Trainer as SetFitTrainer
from setfit import TrainingArguments as SetFitArgs

import datasets

from ml.training.evaluate import compute_metrics, log_metrics_to_mlflow
from ml.utils.reproducibility import SEED, set_seed

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "fewshot"
MODEL_NAME = "answerdotai/ModernBERT-base"
SHOT_COUNTS = [2, 4, 8, 16, 32]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_label_names() -> list[str]:
    """Return Banking77 canonical label names."""
    from ml.training.train import LABEL_NAMES
    return LABEL_NAMES


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test parquet splits."""
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")
    logger.info(f"Loaded train ({len(train_df)}) and test ({len(test_df)})")
    return train_df, test_df


def sample_few_shot(df: pd.DataFrame, n_shots: int, seed: int = SEED) -> pd.DataFrame:
    """Sample n_shots examples per class from a DataFrame."""
    sampled = df.groupby("label", group_keys=False).apply(lambda g: g.sample(n=min(n_shots, len(g)), random_state=seed))
    logger.info(f"Sampled {len(sampled)} examples ({n_shots} shots/class)")
    return sampled


def plot_learning_curve(results: list[dict], save_path: Path) -> None:
    """Plot shots vs accuracy and F1 macro."""
    shots = [r["shots"] for r in results]
    accuracy = [r["accuracy"] for r in results]
    f1_macro = [r["f1_macro"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(shots, accuracy, "o-", label="Accuracy", linewidth=2)
    ax.plot(shots, f1_macro, "s-", label="F1 (macro)", linewidth=2)
    ax.set_xlabel("Shots per class")
    ax.set_ylabel("Score")
    ax.set_title("SetFit Few-Shot Learning Curve (Banking77)")
    ax.set_xticks(shots)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Learning curve saved to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    set_seed(SEED)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    label_names = load_label_names()
    num_labels = len(label_names)
    train_df, test_df = load_splits()

    # Convert to HF datasets for SetFit
    test_ds = datasets.Dataset.from_dict(
        {
            "text": test_df["text"].tolist(),
            "label": test_df["label"].tolist(),
        }
    )

    mlflow.set_experiment("banking77-fewshot")
    all_results: list[dict] = []

    with mlflow.start_run(run_name="setfit-learning-curve"):
        mlflow.log_params(
            {
                "model": MODEL_NAME,
                "shot_counts": str(SHOT_COUNTS),
                "num_labels": num_labels,
                "seed": SEED,
            }
        )

        for n_shots in SHOT_COUNTS:
            logger.info(f"--- {n_shots}-shot experiment ---")

            with mlflow.start_run(run_name=f"setfit-{n_shots}shot", nested=True):
                # Sample training data
                few_df = sample_few_shot(train_df, n_shots)
                train_ds = datasets.Dataset.from_dict(
                    {
                        "text": few_df["text"].tolist(),
                        "label": few_df["label"].tolist(),
                    }
                )

                # SetFit model
                model = SetFitModel.from_pretrained(
                    MODEL_NAME,
                    labels=label_names,
                )

                # Training args
                args = SetFitArgs(
                    output_dir=str(ARTIFACTS_DIR / f"setfit-{n_shots}shot"),
                    num_epochs=1,
                    batch_size=16,
                    seed=SEED,
                )

                trainer = SetFitTrainer(
                    model=model,
                    args=args,
                    train_dataset=train_ds,
                    eval_dataset=test_ds,
                )

                # Train
                trainer.train()

                # Evaluate on full test set
                y_pred = model.predict(test_df["text"].tolist())
                y_pred = np.array(y_pred)
                y_true = test_df["label"].values

                metrics = compute_metrics(y_true, y_pred)

                # Log
                mlflow.log_params(
                    {
                        "shots_per_class": n_shots,
                        "train_samples": len(few_df),
                    }
                )
                log_metrics_to_mlflow(metrics)

                result = {"shots": n_shots, **metrics}
                all_results.append(result)

                logger.success(
                    f"{n_shots}-shot -> Accuracy: {metrics['accuracy']:.4f}  F1 (macro): {metrics['f1_macro']:.4f}"
                )

        # Save learning curve data
        results_df = pd.DataFrame(all_results)
        csv_path = ARTIFACTS_DIR / "learning_curve_data.csv"
        results_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(str(csv_path))

        # Plot learning curve
        plot_path = ARTIFACTS_DIR / "learning_curve.png"
        plot_learning_curve(all_results, plot_path)
        mlflow.log_artifact(str(plot_path))

        # Log best result at parent level
        best = max(all_results, key=lambda r: r["f1_macro"])
        mlflow.log_metric("best_f1_macro", best["f1_macro"])
        mlflow.log_metric("best_shots", best["shots"])

        logger.info("\n--- Few-Shot Results Summary ---")
        for r in all_results:
            logger.info(f"  {r['shots']:2d}-shot: Acc={r['accuracy']:.4f}  F1={r['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
