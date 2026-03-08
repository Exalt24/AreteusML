"""Optuna hyperparameter search for ModernBERT on Banking77.

Run with:
    python -m ml.training.hyperparameter_search
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
import torch
from loguru import logger
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
)

from ml.training.evaluate import log_metrics_to_mlflow
from ml.training.train import Banking77Dataset, WeightedTrainer, make_hf_compute_metrics
from ml.utils.reproducibility import SEED, set_seed

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "hpsearch"
MODEL_NAME = "answerdotai/ModernBERT-base"
N_TRIALS = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_label_names() -> list[str]:
    from ml.training.train import LABEL_NAMES
    return LABEL_NAMES


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splits = {}
    for name in ("train", "val", "test"):
        splits[name] = pd.read_parquet(DATA_DIR / f"{name}.parquet")
    return splits["train"], splits["val"], splits["test"]


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
def objective(trial: optuna.Trial) -> float:
    """Single Optuna trial: train ModernBERT with sampled hyperparameters."""
    # Sample hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16])
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.2)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1)
    num_epochs = trial.suggest_categorical("num_epochs", [5, 8, 10])

    grad_accum = 32 // batch_size  # Keep effective batch size ~32

    logger.info(
        f"Trial {trial.number}: lr={lr:.2e}, bs={batch_size}, "
        f"warmup={warmup_ratio:.2f}, wd={weight_decay:.3f}, epochs={num_epochs}"
    )

    with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
        mlflow.log_params(
            {
                "trial": trial.number,
                "learning_rate": lr,
                "batch_size": batch_size,
                "gradient_accumulation_steps": grad_accum,
                "warmup_ratio": warmup_ratio,
                "weight_decay": weight_decay,
                "num_epochs": num_epochs,
            }
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

        output_dir = str(ARTIFACTS_DIR / f"trial-{trial.number}")
        fp16 = torch.cuda.is_available()

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=32,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            fp16=fp16,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1_macro",
            greater_is_better=True,
            save_total_limit=1,
            seed=SEED,
            report_to="none",  # We log manually via MLflow
        )

        trainer = WeightedTrainer(
            class_weights=CLASS_WEIGHTS_TENSOR,
            model=model,
            args=training_args,
            train_dataset=TRAIN_DATASET,
            eval_dataset=VAL_DATASET,
            processing_class=TOKENIZER,
            compute_metrics=make_hf_compute_metrics(LABEL_NAMES),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        trainer.train()

        # Evaluate best checkpoint on validation set
        eval_result = trainer.evaluate()
        f1_macro = eval_result["eval_f1_macro"]

        log_metrics_to_mlflow(eval_result)
        logger.info(f"Trial {trial.number} -> eval_f1_macro: {f1_macro:.4f}")

    return f1_macro


# ---------------------------------------------------------------------------
# Module-level state (populated in main, used by objective)
# ---------------------------------------------------------------------------
LABEL_NAMES: list[str] = []
NUM_LABELS: int = 0
ID2LABEL: dict = {}
LABEL2ID: dict = {}
CLASS_WEIGHTS_TENSOR: torch.Tensor = torch.tensor([])
TOKENIZER = None
TRAIN_DATASET = None
VAL_DATASET = None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    global LABEL_NAMES, NUM_LABELS, ID2LABEL, LABEL2ID
    global CLASS_WEIGHTS_TENSOR, TOKENIZER, TRAIN_DATASET, VAL_DATASET

    set_seed(SEED)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    LABEL_NAMES = load_label_names()
    NUM_LABELS = len(LABEL_NAMES)
    ID2LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}
    LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}

    train_df, val_df, test_df = load_splits()
    logger.info(f"Classes: {NUM_LABELS} | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Class weights
    class_weights = compute_class_weight(
        "balanced",
        classes=np.arange(NUM_LABELS),
        y=train_df["label"].values,
    )
    CLASS_WEIGHTS_TENSOR = torch.tensor(class_weights, dtype=torch.float32)

    # Tokenize once
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_enc = TOKENIZER(train_df["text"].tolist(), truncation=True, padding=True, max_length=128)
    val_enc = TOKENIZER(val_df["text"].tolist(), truncation=True, padding=True, max_length=128)

    TRAIN_DATASET = Banking77Dataset(train_enc, train_df["label"].tolist())
    VAL_DATASET = Banking77Dataset(val_enc, val_df["label"].tolist())

    # Optuna study
    mlflow.set_experiment("banking77-hpsearch")

    with mlflow.start_run(run_name="optuna-hpsearch"):
        mlflow.log_params(
            {
                "model": MODEL_NAME,
                "n_trials": N_TRIALS,
                "seed": SEED,
                "optimize_metric": "eval_f1_macro",
            }
        )

        study = optuna.create_study(
            direction="maximize",
            study_name="banking77-modernbert-hpsearch",
            sampler=optuna.samplers.TPESampler(seed=SEED),
        )
        study.optimize(objective, n_trials=N_TRIALS)

        # Log best trial
        best = study.best_trial
        logger.success(f"Best trial #{best.number}: f1_macro={best.value:.4f}")
        logger.info(f"Best params: {best.params}")

        mlflow.log_metric("best_f1_macro", best.value)
        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})

        # Save study results
        results_df = study.trials_dataframe()
        csv_path = ARTIFACTS_DIR / "optuna_results.csv"
        results_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(str(csv_path))

        # Optimization history plot
        try:
            fig = optuna.visualization.matplotlib.plot_optimization_history(study)
            plot_path = ARTIFACTS_DIR / "optimization_history.png"
            fig.figure.savefig(plot_path, dpi=150, bbox_inches="tight")
            mlflow.log_artifact(str(plot_path))
        except Exception:
            logger.warning("Could not generate optimization history plot")

        # Param importances plot
        try:
            fig = optuna.visualization.matplotlib.plot_param_importances(study)
            plot_path = ARTIFACTS_DIR / "param_importances.png"
            fig.figure.savefig(plot_path, dpi=150, bbox_inches="tight")
            mlflow.log_artifact(str(plot_path))
        except Exception:
            logger.warning("Could not generate param importances plot")


if __name__ == "__main__":
    main()
