"""Fine-tune ModernBERT-base on Banking77 with class-weighted loss.

Run with:
    python -m ml.training.train
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

import matplotlib.pyplot as plt

from ml.training.evaluate import (
    compute_metrics,
    log_metrics_to_mlflow,
    per_class_report,
    plot_confusion_matrix,
)
from ml.training.labels import LABEL_NAMES, load_label_names  # noqa: F401
from ml.utils.reproducibility import SEED, cleanup_mlflow_nulls, get_device, set_seed

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "ml" / "models" / "production"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "modernbert"
MODEL_NAME = "answerdotai/ModernBERT-base"
MAX_LENGTH = 64  # Banking77 queries avg ~12 words; 64 tokens is plenty


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class Banking77Dataset(torch.utils.data.Dataset):
    """Simple map-style dataset for tokenized Banking77 data."""

    def __init__(self, encodings: dict, labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ---------------------------------------------------------------------------
# Weighted Trainer
# ---------------------------------------------------------------------------
class WeightedTrainer(Trainer):
    """Trainer subclass that applies class weights to cross-entropy loss."""

    def __init__(self, class_weights: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self._loss_fn: torch.nn.CrossEntropyLoss | None = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self._loss_fn is None or self._loss_fn.weight.device != logits.device:
            self._loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = self._loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# LABEL_NAMES and load_label_names are re-exported from ml.training.labels
# (imported above) for backward compatibility.


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, val, and test parquet splits."""
    splits = {}
    for name in ("train", "val", "test"):
        path = DATA_DIR / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run the data pipeline first.")
        splits[name] = pd.read_parquet(path)
        logger.info(f"Loaded {name}: {len(splits[name])} samples")
    return splits["train"], splits["val"], splits["test"]


def make_hf_compute_metrics(label_names: list[str]):
    """Return a compute_metrics function compatible with HF Trainer."""

    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        metrics = compute_metrics(labels, preds)
        return metrics

    return _compute_metrics


# ---------------------------------------------------------------------------
# Pipeline-callable functions (used by Dagster training_pipeline)
# ---------------------------------------------------------------------------
def train_modernbert(train_df: pd.DataFrame, output_dir: str) -> str:
    """Train ModernBERT on the given DataFrame. Returns path to saved model."""
    set_seed(SEED)
    label_names = load_label_names()
    num_labels = len(label_names)

    class_weights = compute_class_weight("balanced", classes=np.arange(num_labels), y=train_df["label"].values)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_enc = tokenizer(train_df["text"].tolist(), truncation=True, padding=True, max_length=MAX_LENGTH)
    train_dataset = Banking77Dataset(train_enc, train_df["label"].tolist())

    id2label = dict(enumerate(label_names))
    label2id = {name: i for i, name in enumerate(label_names)}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels, id2label=id2label, label2id=label2id
    )

    fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=fp16,
        gradient_checkpointing=True,
        optim="adafactor",
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=1,
        save_only_model=True,
        seed=SEED,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    save_dir = str(Path(output_dir) / "final")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    logger.info(f"Pipeline model saved to {save_dir}")
    return save_dir


def load_model_and_predict(model_path: str, texts: list[str]) -> list[int]:
    """Load a saved model and return predicted label indices."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encodings)
    return np.argmax(outputs.logits.numpy(), axis=-1).tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    set_seed(SEED)
    get_device()
    cleaned = cleanup_mlflow_nulls()
    if cleaned:
        logger.info(f"Cleaned {cleaned} corrupted MLflow metric file(s) from previous interrupted run")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load data and labels
    label_names = load_label_names()
    num_labels = len(label_names)
    train_df, val_df, test_df = load_splits()

    logger.info(f"Classes: {num_labels} | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Compute class weights
    class_weights = compute_class_weight(
        "balanced",
        classes=np.arange(num_labels),
        y=train_df["label"].values,
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    logger.info(f"Class weight range: [{class_weights.min():.3f}, {class_weights.max():.3f}]")

    # Tokenize (defer test set until after training to save RAM)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_enc = tokenizer(train_df["text"].tolist(), truncation=True, padding=True, max_length=MAX_LENGTH)
    val_enc = tokenizer(val_df["text"].tolist(), truncation=True, padding=True, max_length=MAX_LENGTH)

    train_dataset = Banking77Dataset(train_enc, train_df["label"].tolist())
    val_dataset = Banking77Dataset(val_enc, val_df["label"].tolist())

    # Free raw DataFrames and encodings (data is now in Dataset objects)
    del train_enc, val_enc, train_df, val_df
    gc.collect()

    # Model
    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # Training arguments
    fp16 = torch.cuda.is_available()
    output_dir = str(ARTIFACTS_DIR / "checkpoints")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=fp16,
        gradient_checkpointing=True,
        optim="adafactor",  # ~1.5GB optimizer states vs ~3GB for AdamW
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        save_only_model=True,
        seed=SEED,
        report_to="mlflow",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,  # Save RAM on Windows
    )

    # Free VRAM fragmentation before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Trainer
    model.gradient_checkpointing_enable()
    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=make_hf_compute_metrics(label_names),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # MLflow
    mlflow.set_experiment("banking77-modernbert")

    run_name = f"modernbert-finetune-{int(time.time())}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model": MODEL_NAME,
                "num_labels": num_labels,
                "epochs": 10,
                "batch_size": 2,
                "gradient_accumulation_steps": 16,
                "optim": "adafactor",
                "effective_batch_size": 32,
                "learning_rate": 2e-5,
                "lr_scheduler": "cosine",
                "warmup_ratio": 0.1,
                "weight_decay": 0.01,
                "fp16": fp16,
                "early_stopping_patience": 3,
                "seed": SEED,
                "class_weighted_loss": True,
            }
        )

        # Log VRAM usage before training
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"VRAM before training: {alloc:.2f} GB allocated, {reserved:.2f} GB reserved")

        # Train
        logger.info("Starting training...")
        trainer.train()
        logger.success("Training complete.")

        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1024**3
            logger.info(f"Peak VRAM during training: {peak:.2f} GB")

        # Save best model
        trainer.save_model(str(MODEL_DIR))
        tokenizer.save_pretrained(str(MODEL_DIR))
        logger.info(f"Best model saved to {MODEL_DIR}")

        # Evaluate on test set (tokenize now to save RAM during training)
        logger.info("Evaluating on test set...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        test_enc = tokenizer(test_df["text"].tolist(), truncation=True, padding=True, max_length=MAX_LENGTH)
        test_dataset = Banking77Dataset(test_enc, test_df["label"].tolist())
        test_preds = trainer.predict(test_dataset)
        y_pred = np.argmax(test_preds.predictions, axis=-1)
        y_true = test_df["label"].values
        y_prob = torch.softmax(torch.tensor(test_preds.predictions), dim=-1).numpy()

        test_metrics = compute_metrics(y_true, y_pred, y_prob)

        # Prefix test metrics and log
        test_metrics_prefixed = {f"test_{k}": v for k, v in test_metrics.items()}
        log_metrics_to_mlflow(test_metrics_prefixed)

        logger.success(
            f"Test Accuracy: {test_metrics['accuracy']:.4f}  Test F1 (macro): {test_metrics['f1_macro']:.4f}"
        )

        # Confusion matrix
        cm_path = ARTIFACTS_DIR / "test_confusion_matrix.png"
        plot_confusion_matrix(y_true, y_pred, label_names, save_path=cm_path)
        plt.close("all")  # Free matplotlib memory
        mlflow.log_artifact(str(cm_path))

        # Per-class report
        report_df = per_class_report(y_true, y_pred, label_names)
        report_path = ARTIFACTS_DIR / "test_per_class_report.csv"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(report_path)
        mlflow.log_artifact(str(report_path))

        # Free test evaluation objects
        del test_preds, test_enc, test_dataset, y_prob
        gc.collect()

        logger.info("All artifacts logged to MLflow.")

    # Cleanup: remove checkpoints to free disk space (best model already saved to MODEL_DIR)
    import shutil

    checkpoints_dir = ARTIFACTS_DIR / "checkpoints"
    if checkpoints_dir.exists():
        shutil.rmtree(checkpoints_dir)
        logger.info("Cleaned up training checkpoints to free disk space")

    # Free GPU memory and all remaining large objects
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Freed GPU memory")


if __name__ == "__main__":
    main()
