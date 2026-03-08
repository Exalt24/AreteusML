"""Tests for data loading and splitting pipeline."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def banking77_df():
    """Full-sized Banking77-like DataFrame."""
    np.random.seed(42)
    n_per_class = 20
    n_classes = 77
    texts = [f"Banking query {i} for class {c}" for c in range(n_classes) for i in range(n_per_class)]
    labels = np.repeat(np.arange(n_classes), n_per_class).tolist()
    return pd.DataFrame({"text": texts, "label": labels})


class TestLoadBanking77:
    def test_load_banking77(self):
        """Test load_banking77 returns DataFrame with correct columns."""
        mock_dataset = {
            "train": MagicMock(),
            "test": MagicMock(),
        }
        mock_dataset["train"].to_pandas.return_value = pd.DataFrame(
            {
                "text": ["query 1", "query 2"],
                "label": [0, 1],
            }
        )
        mock_dataset["test"].to_pandas.return_value = pd.DataFrame(
            {
                "text": ["query 3"],
                "label": [2],
            }
        )

        with patch("ml.data.load_data.load_dataset", return_value=mock_dataset):
            from ml.data.load_data import load_banking77

            df = load_banking77()
            assert "text" in df.columns
            assert "label" in df.columns
            assert len(df) == 3


class TestStratifiedSplit:
    def test_stratified_split_proportions(self, banking77_df):
        with patch("ml.data.load_data.set_seed"):
            from ml.data.load_data import stratified_split

            train, val, test = stratified_split(banking77_df)
            total = len(banking77_df)
            assert abs(len(train) / total - 0.70) < 0.05
            assert abs(len(val) / total - 0.15) < 0.05
            assert abs(len(test) / total - 0.15) < 0.05

    def test_no_data_leakage_between_splits(self, banking77_df):
        with patch("ml.data.load_data.set_seed"):
            from ml.data.load_data import stratified_split

            train, val, test = stratified_split(banking77_df)
            train_idx = set(train.index)
            val_idx = set(val.index)
            test_idx = set(test.index)
            assert train_idx.isdisjoint(val_idx)
            assert train_idx.isdisjoint(test_idx)
            assert val_idx.isdisjoint(test_idx)

    def test_all_classes_in_all_splits(self, banking77_df):
        with patch("ml.data.load_data.set_seed"):
            from ml.data.load_data import stratified_split

            train, val, test = stratified_split(banking77_df)
            all_classes = set(banking77_df["label"].unique())
            assert set(train["label"].unique()) == all_classes
            assert set(val["label"].unique()) == all_classes
            assert set(test["label"].unique()) == all_classes

    def test_text_column_no_nulls(self, banking77_df):
        with patch("ml.data.load_data.set_seed"):
            from ml.data.load_data import stratified_split

            train, val, test = stratified_split(banking77_df)
            for split in [train, val, test]:
                assert split["text"].isnull().sum() == 0

    def test_label_range(self, banking77_df):
        with patch("ml.data.load_data.set_seed"):
            from ml.data.load_data import stratified_split

            train, val, test = stratified_split(banking77_df)
            for split in [train, val, test]:
                assert split["label"].min() >= 0
                assert split["label"].max() <= 76

    def test_text_not_empty(self, banking77_df):
        with patch("ml.data.load_data.set_seed"):
            from ml.data.load_data import stratified_split

            train, _, _ = stratified_split(banking77_df)
            assert (train["text"].str.len() > 0).all()

    def test_reproducible_split(self, banking77_df):
        with patch("ml.data.load_data.set_seed"):
            from ml.data.load_data import stratified_split

            train1, val1, test1 = stratified_split(banking77_df)
            train2, val2, test2 = stratified_split(banking77_df)
            pd.testing.assert_frame_equal(train1, train2)
            pd.testing.assert_frame_equal(val1, val2)
            pd.testing.assert_frame_equal(test1, test2)
