"""Tests for feedback endpoints."""

from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture
def _mock_feedback():
    with (
        patch("backend.app.api.routes.feedback.log_feedback", new_callable=AsyncMock) as mock_log,
        patch("backend.app.api.routes.feedback.get_feedback_stats", new_callable=AsyncMock) as mock_stats,
    ):
        mock_log.return_value = "abc123feedbackid"
        mock_stats.return_value = {
            "total_corrections": 42,
            "top_corrected_classes": [
                {"label": 5, "label_name": "activate_my_card", "count": 10},
            ],
        }
        yield mock_log, mock_stats


@pytest.fixture
def _mock_feedback_empty_stats():
    with patch("backend.app.api.routes.feedback.get_feedback_stats", new_callable=AsyncMock) as mock_stats:
        mock_stats.return_value = {
            "total_corrections": 0,
            "top_corrected_classes": [],
        }
        yield


class TestSubmitFeedback:
    def test_submit_feedback(self, test_client, _mock_feedback):
        resp = test_client.post(
            "/feedback",
            json={
                "prediction_id": "pred_001",
                "correct_label": 5,
                "correct_label_name": "activate_my_card",
                "comment": "Wrong prediction",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "recorded"
        assert "feedback_id" in data

    def test_feedback_invalid_label(self, test_client, _mock_feedback):
        resp = test_client.post(
            "/feedback",
            json={
                "prediction_id": "pred_001",
                "correct_label": -1,
            },
        )
        assert resp.status_code == 422

    def test_feedback_missing_fields(self, test_client, _mock_feedback):
        resp = test_client.post("/feedback", json={})
        assert resp.status_code == 422

    def test_submit_feedback_nonexistent_prediction(self, test_client, _mock_feedback):
        # The endpoint does not validate prediction_id existence, it just logs
        resp = test_client.post(
            "/feedback",
            json={
                "prediction_id": "nonexistent_pred_999",
                "correct_label": 0,
            },
        )
        assert resp.status_code == 200


class TestFeedbackStats:
    def test_feedback_stats(self, test_client, _mock_feedback):
        resp = test_client.get("/feedback/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_corrections"] == 42
        assert len(data["top_corrected_classes"]) > 0

    def test_feedback_stats_empty(self, test_client, _mock_feedback_empty_stats):
        resp = test_client.get("/feedback/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_corrections"] == 0
        assert data["top_corrected_classes"] == []
