"""Pandera schema validation for Banking77 dataset."""

import pandas as pd
import pandera as pa
from loguru import logger

from ml.utils.reproducibility import SEED, set_seed

NUM_CLASSES = 77
MIN_SAMPLES_PER_CLASS = 10

banking77_schema = pa.DataFrameSchema(
    columns={
        "text": pa.Column(
            str,
            nullable=False,
            checks=[
                pa.Check.str_length(min_value=1, max_value=512),
            ],
        ),
        "label": pa.Column(
            int,
            nullable=False,
            checks=[
                pa.Check.in_range(min_value=0, max_value=76),
            ],
        ),
    },
    strict=False,
    coerce=True,
)


def validate_schema(df: pd.DataFrame, split_name: str = "unknown") -> pd.DataFrame:
    """Validate a DataFrame against the Banking77 schema.

    Returns the validated DataFrame or raises a SchemaError.
    """
    logger.info(f"Validating schema for '{split_name}' split ({len(df)} rows)...")
    validated = banking77_schema.validate(df)
    logger.success(f"Schema validation passed for '{split_name}' split.")
    return validated


def check_class_distribution(df: pd.DataFrame, split_name: str = "unknown") -> bool:
    """Check that every class has at least MIN_SAMPLES_PER_CLASS samples.

    Returns True if all classes pass, False otherwise.
    """
    set_seed(SEED)
    logger.info(f"Checking class distribution for '{split_name}' split...")

    class_counts = df["label"].value_counts()
    num_classes_present = len(class_counts)

    if num_classes_present < NUM_CLASSES:
        logger.warning(f"[{split_name}] Only {num_classes_present}/{NUM_CLASSES} classes present.")

    low_count_classes = class_counts[class_counts < MIN_SAMPLES_PER_CLASS]
    if len(low_count_classes) > 0:
        for label, count in low_count_classes.items():
            logger.warning(
                f"[{split_name}] Class {label} has only {count} samples (minimum recommended: {MIN_SAMPLES_PER_CLASS})."
            )
        return False

    logger.success(
        f"[{split_name}] All {num_classes_present} classes have >= {MIN_SAMPLES_PER_CLASS} samples. "
        f"Min: {class_counts.min()}, Max: {class_counts.max()}"
    )
    return True


def validate_split(df: pd.DataFrame, split_name: str = "unknown") -> pd.DataFrame:
    """Run full validation: schema + distribution check."""
    validated = validate_schema(df, split_name)
    check_class_distribution(validated, split_name)
    return validated
