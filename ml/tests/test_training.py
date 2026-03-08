"""Tests for training evaluation utilities."""

import numpy as np
import pandas as pd


class TestComputeMetrics:
    def test_compute_metrics_format(self, y_true, y_pred):
        from ml.training.evaluate import compute_metrics

        metrics = compute_metrics(y_true, y_pred)
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "f1_micro" in metrics
        assert "f1_weighted" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics

    def test_compute_metrics_perfect(self, y_true, y_pred):
        from ml.training.evaluate import compute_metrics

        # y_pred fixture is identical to y_true
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0

    def test_compute_metrics_random(self, y_true, y_pred_random):
        from ml.training.evaluate import compute_metrics

        metrics = compute_metrics(y_true, y_pred_random)
        assert 0 <= metrics["accuracy"] <= 1.0
        assert 0 <= metrics["f1_macro"] <= 1.0


class TestPerClassReport:
    def test_per_class_report_shape(self):
        from ml.training.evaluate import per_class_report

        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 1])
        labels = ["class_0", "class_1", "class_2"]

        report = per_class_report(y_true, y_pred, labels)
        assert isinstance(report, pd.DataFrame)
        assert len(report) == 3
        assert "precision" in report.columns
        assert "recall" in report.columns
        assert "f1-score" in report.columns


class TestConfusionMatrix:
    def test_confusion_matrix_shape(self):
        from sklearn.metrics import confusion_matrix

        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 1])
        cm = confusion_matrix(y_true, y_pred)
        assert cm.shape == (3, 3)
        # Diagonal should have correct predictions
        assert cm[0, 0] == 2  # class 0 predicted correctly twice
