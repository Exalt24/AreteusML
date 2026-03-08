"""Load Banking77 dataset from HuggingFace and create stratified splits."""

from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from ml.utils.reproducibility import SEED, set_seed

TRAIN_URL = "https://huggingface.co/datasets/PolyAI/banking77/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
TEST_URL = "https://huggingface.co/datasets/PolyAI/banking77/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet"

PROCESSED_DIR = Path(__file__).resolve().parent / "processed"


def load_banking77() -> pd.DataFrame:
    """Load the Banking77 dataset from HuggingFace parquet export."""
    logger.info("Loading Banking77 dataset from HuggingFace parquet export...")
    train_df = pd.read_parquet(TRAIN_URL)
    test_df = pd.read_parquet(TEST_URL)
    combined = pd.concat([train_df, test_df], ignore_index=True)
    logger.info(f"Loaded {len(combined)} total samples with {combined['label'].nunique()} classes")
    return combined


def stratified_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform stratified 70/15/15 train/val/test split."""
    set_seed(SEED)

    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED)

    logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Save split DataFrames as parquet files."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = PROCESSED_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
        logger.info(f"Saved {name} split ({len(df)} samples) to {path}")


def main() -> None:
    """Load Banking77, split, and save to processed directory."""
    df = load_banking77()
    train_df, val_df, test_df = stratified_split(df)
    save_splits(train_df, val_df, test_df)
    logger.success("Data loading and splitting complete.")


if __name__ == "__main__":
    main()
