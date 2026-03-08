"""Dagster training pipeline assets for Banking77 intent classification."""

import json
import random
from pathlib import Path

import mlflow
import pandas as pd
from dagster import AssetExecutionContext, ConfigurableResource, asset
from loguru import logger
from pandera import Check, Column, DataFrameSchema

# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


class MLflowResource(ConfigurableResource):
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "banking77-modernbert"


class ModelPathResource(ConfigurableResource):
    base_dir: str = "artifacts/models"
    data_dir: str = "artifacts/data"


# ---------------------------------------------------------------------------
# Helpers - lightweight data augmentation
# ---------------------------------------------------------------------------


def _synonym_replace(text: str, n: int = 1) -> str:
    """Naive synonym replacement: swap random words with a simple synonym map."""
    synonym_map = {
        "money": "funds",
        "transfer": "send",
        "account": "profile",
        "card": "payment card",
        "bank": "financial institution",
        "pay": "remit",
        "cash": "money",
        "check": "verify",
        "lost": "missing",
        "stolen": "taken",
        "block": "freeze",
        "pin": "security code",
        "fee": "charge",
        "balance": "amount",
        "deposit": "credit",
        "withdraw": "debit",
    }
    words = text.split()
    replaced = 0
    for i, w in enumerate(words):
        lower = w.lower()
        if lower in synonym_map and replaced < n:
            words[i] = synonym_map[lower]
            replaced += 1
    return " ".join(words)


def _random_swap(text: str, n: int = 1) -> str:
    """Randomly swap two words in the sentence."""
    words = text.split()
    if len(words) < 2:
        return text
    for _ in range(n):
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return " ".join(words)


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------


@asset(group_name="training")
def raw_data(context: AssetExecutionContext, model_paths: ModelPathResource) -> pd.DataFrame:
    """Load Banking77 dataset and save as parquet."""
    logger.info("Loading Banking77 dataset from HuggingFace parquet export")
    train_url = "https://huggingface.co/datasets/PolyAI/banking77/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
    test_url = "https://huggingface.co/datasets/PolyAI/banking77/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet"
    train_df = pd.read_parquet(train_url)
    test_df = pd.read_parquet(test_url)

    data_dir = Path(model_paths.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "raw_train.parquet"
    test_path = data_dir / "raw_test.parquet"
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    logger.info(f"Saved {len(train_df)} train and {len(test_df)} test samples to {data_dir}")
    context.log.info(f"Raw data: {len(train_df)} train, {len(test_df)} test")
    return train_df


@asset(group_name="training")
def validated_data(context: AssetExecutionContext, raw_data: pd.DataFrame) -> pd.DataFrame:
    """Run Pandera schema validation on raw data."""
    schema = DataFrameSchema(
        {
            "text": Column(str, Check(lambda s: s.str.len() > 0, error="Empty text")),
            "label": Column(int, Check.in_range(0, 76)),
        },
        coerce=True,
    )

    validated = schema.validate(raw_data)
    logger.info(f"Validation passed for {len(validated)} rows")
    context.log.info(f"Validated {len(validated)} rows")
    return validated


@asset(group_name="training")
def augmented_data(
    context: AssetExecutionContext,
    validated_data: pd.DataFrame,
    model_paths: ModelPathResource,
) -> pd.DataFrame:
    """Augment underrepresented classes via synonym replacement and random swap."""
    label_counts = validated_data["label"].value_counts()
    median_count = int(label_counts.median())

    logger.info(f"Median class count: {median_count}")

    augmented_rows: list[dict] = []
    for label, count in label_counts.items():
        if count >= median_count:
            continue
        subset = validated_data[validated_data["label"] == label]
        needed = median_count - count
        for _ in range(needed):
            row = subset.sample(1).iloc[0]
            text = row["text"]
            new_text = _synonym_replace(text) if random.random() < 0.5 else _random_swap(text)  # noqa: S311
            augmented_rows.append({"text": new_text, "label": label})

    aug_df = pd.DataFrame(augmented_rows)
    combined = pd.concat([validated_data, aug_df], ignore_index=True)

    data_dir = Path(model_paths.data_dir)
    combined.to_parquet(data_dir / "augmented_train.parquet", index=False)

    logger.info(f"Augmented {len(augmented_rows)} samples; total: {len(combined)}")
    context.log.info(f"Augmented data: {len(combined)} total rows (+{len(augmented_rows)} new)")
    return combined


@asset(group_name="training")
def trained_model(
    context: AssetExecutionContext,
    augmented_data: pd.DataFrame,
    mlflow_resource: MLflowResource,
    model_paths: ModelPathResource,
) -> str:
    """Fine-tune ModernBERT on the augmented training data."""
    from ml.training.train import train_modernbert  # local import to avoid top-level heavy deps

    mlflow.set_tracking_uri(mlflow_resource.tracking_uri)
    mlflow.set_experiment(mlflow_resource.experiment_name)

    output_dir = Path(model_paths.base_dir) / "modernbert-banking77"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting ModernBERT fine-tuning with {len(augmented_data)} samples")

    with mlflow.start_run(run_name="training-pipeline"):
        mlflow.log_param("train_samples", len(augmented_data))
        model_path = train_modernbert(
            train_df=augmented_data,
            output_dir=str(output_dir),
        )
        mlflow.log_param("model_path", model_path)

    logger.info(f"Model saved to {model_path}")
    context.log.info(f"Trained model at {model_path}")
    return str(model_path)


@asset(group_name="training")
def existing_model(context: AssetExecutionContext) -> str:
    """Return path to the existing production model, skipping training."""
    model_path = "ml/models/production"
    logger.info(f"Using existing production model at {model_path}")
    context.log.info(f"Existing model path: {model_path}")
    return model_path


@asset(group_name="training")
def evaluated_model(
    context: AssetExecutionContext,
    existing_model: str,
    mlflow_resource: MLflowResource,
    model_paths: ModelPathResource,
) -> dict:
    """Load pre-computed test metrics for the existing model."""
    mlflow.set_tracking_uri(mlflow_resource.tracking_uri)
    mlflow.set_experiment(mlflow_resource.experiment_name)

    metrics_path = Path("artifacts/modernbert/test_metrics.json")
    logger.info(f"Loading metrics from {metrics_path} for model at {existing_model}")

    with open(metrics_path) as f:
        metrics = json.load(f)

    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, int | float)})

    logger.info(f"Evaluation metrics: {metrics}")
    context.log.info(f"Loaded metrics from {metrics_path}")

    return metrics


@asset(group_name="training")
def registered_model(
    context: AssetExecutionContext,
    existing_model: str,
    evaluated_model: dict,
    mlflow_resource: MLflowResource,
) -> str:
    """Register model in MLflow if metrics meet threshold (F1 > 0.90)."""
    f1 = evaluated_model.get("f1_weighted", 0.0)
    threshold = 0.90
    model_name = "banking77-modernbert"

    mlflow.set_tracking_uri(mlflow_resource.tracking_uri)

    if f1 < threshold:
        msg = f"F1 {f1:.4f} below threshold {threshold}. Model NOT registered."
        logger.warning(msg)
        context.log.warning(msg)
        return "NOT_REGISTERED"

    logger.info(f"F1 {f1:.4f} meets threshold {threshold}. Registering model.")

    with mlflow.start_run(run_name="registration"):
        mlflow.log_metrics(evaluated_model)
        model_uri = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=None,
            artifacts={"model_dir": existing_model},
        ).model_uri

    result = mlflow.register_model(model_uri, model_name)
    logger.info(f"Registered model version {result.version} as '{model_name}'")
    context.log.info(f"Registered {model_name} v{result.version}")
    return f"{model_name}/v{result.version}"
