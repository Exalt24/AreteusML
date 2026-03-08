"""Shared fixtures for ML tests."""

import numpy as np
import pandas as pd
import pytest

NUM_CLASSES = 77


@pytest.fixture
def label_names():
    return [f"class_{i}" for i in range(NUM_CLASSES)]


@pytest.fixture
def sample_dataframe():
    """Banking77-like DataFrame with text and label columns."""
    np.random.seed(42)
    n_samples = 770  # 10 per class
    texts = [f"Sample banking query number {i}" for i in range(n_samples)]
    labels = np.repeat(np.arange(NUM_CLASSES), 10).tolist()
    return pd.DataFrame({"text": texts, "label": labels})


@pytest.fixture
def sample_texts():
    return [
        "I need to activate my new card",
        "How do I check my balance?",
        "I want to transfer money",
        "What is my pin number?",
        "Cancel my transaction",
    ]


@pytest.fixture
def y_true():
    return np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])


@pytest.fixture
def y_pred():
    return np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])


@pytest.fixture
def y_pred_random():
    np.random.seed(42)
    return np.random.randint(0, 5, size=10)
