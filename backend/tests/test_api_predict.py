"""Tests for prediction endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def _mock_predict():
    """Patch predict_single and predict_batch with realistic returns."""
    single_result = MagicMock()
    single_result.label = 5
    single_result.confidence = 0.92
    single_result.label_name = "activate_my_card"
    single_result.latency_ms = 12.5

    batch_result = MagicMock()
    batch_result.label = 10
    batch_result.confidence = 0.85
    batch_result.label_name = "check_balance"
    batch_result.latency_ms = 8.3

    with (
        patch(
            "backend.app.api.routes.predict.predict_single", new_callable=AsyncMock, return_value=single_result
        ) as mock_single,
        patch(
            "backend.app.api.routes.predict.predict_batch",
            new_callable=AsyncMock,
            return_value=[batch_result, batch_result],
        ) as mock_batch,
        patch("backend.app.api.routes.predict.get_cached_prediction", new_callable=AsyncMock, return_value=None),
        patch("backend.app.api.routes.predict.cache_prediction", new_callable=AsyncMock),
        patch("backend.app.api.routes.predict.log_prediction"),
    ):
        yield mock_single, mock_batch


@pytest.fixture
def _mock_predict_low_confidence():
    """Patch with low confidence result."""
    result = MagicMock()
    result.label = 10
    result.confidence = 0.3
    result.label_name = "check_balance"
    result.latency_ms = 15.0

    with (
        patch("backend.app.api.routes.predict.predict_single", new_callable=AsyncMock, return_value=result),
        patch("backend.app.api.routes.predict.get_cached_prediction", new_callable=AsyncMock, return_value=None),
        patch("backend.app.api.routes.predict.cache_prediction", new_callable=AsyncMock),
        patch("backend.app.api.routes.predict.log_prediction"),
    ):
        yield


class TestPredictSingle:
    def test_predict_single_success(self, test_client, _mock_predict):
        resp = test_client.post("/predict", json={"text": "activate my card"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] == 5
        assert data["confidence"] == 0.92
        assert data["label_name"] == "activate_my_card"

    def test_predict_single_empty_text(self, test_client):
        resp = test_client.post("/predict", json={"text": ""})
        assert resp.status_code == 422

    def test_predict_single_too_long_text(self, test_client):
        resp = test_client.post("/predict", json={"text": "a" * 10001})
        assert resp.status_code == 422

    def test_predict_response_format(self, test_client, _mock_predict):
        resp = test_client.post("/predict", json={"text": "test query"})
        data = resp.json()
        required_fields = {"label", "confidence", "label_name", "latency_ms", "low_confidence"}
        assert required_fields.issubset(set(data.keys()))

    def test_predict_latency_included(self, test_client, _mock_predict):
        resp = test_client.post("/predict", json={"text": "test query"})
        data = resp.json()
        assert "latency_ms" in data
        assert isinstance(data["latency_ms"], float)
        assert data["latency_ms"] >= 0

    def test_predict_special_characters(self, test_client, _mock_predict):
        resp = test_client.post("/predict", json={"text": "Hello! @#$%^&*() test"})
        assert resp.status_code == 200

    def test_predict_unicode(self, test_client, _mock_predict):
        resp = test_client.post("/predict", json={"text": "Ich moechte mein Konto ueberpruefen"})
        assert resp.status_code == 200

    def test_predict_whitespace_only(self, test_client, _mock_predict):
        resp = test_client.post("/predict", json={"text": "   "})
        # Whitespace-only passes min_length check (3 chars) and is treated as valid input
        assert resp.status_code == 200


class TestPredictLowConfidence:
    def test_predict_low_confidence_flag(self, test_client, _mock_predict_low_confidence):
        resp = test_client.post("/predict", json={"text": "ambiguous query"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["low_confidence"] is True
        assert data["message"] is not None
        assert "human review" in data["message"].lower()


class TestPredictBatch:
    def test_predict_batch_success(self, test_client, _mock_predict):
        resp = test_client.post("/predict/batch", json={"texts": ["query 1", "query 2"]})
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2

    def test_predict_batch_empty_list(self, test_client):
        resp = test_client.post("/predict/batch", json={"texts": []})
        assert resp.status_code == 422

    def test_predict_batch_max_size(self, test_client):
        texts = [f"query {i}" for i in range(101)]
        resp = test_client.post("/predict/batch", json={"texts": texts})
        assert resp.status_code == 422


class TestPredictCaching:
    def test_predict_caching(self, test_client):
        cached_data = {
            "label": 5,
            "confidence": 0.92,
            "label_name": "activate_my_card",
            "latency_ms": 12.5,
            "low_confidence": False,
            "message": None,
        }
        with (
            patch(
                "backend.app.api.routes.predict.get_cached_prediction", new_callable=AsyncMock, return_value=cached_data
            ),
            patch("backend.app.api.routes.predict.cache_prediction", new_callable=AsyncMock),
            patch("backend.app.api.routes.predict.predict_single", new_callable=AsyncMock) as mock_single,
            patch("backend.app.api.routes.predict.log_prediction"),
        ):
            resp = test_client.post("/predict", json={"text": "activate my card"})
            assert resp.status_code == 200
            # predict_single should NOT be called when cache hit
            mock_single.assert_not_called()


class TestHealthAndMetrics:
    def test_health_endpoint(self, test_client):
        resp = test_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_metrics_endpoint(self, test_client):
        resp = test_client.get("/metrics")
        assert resp.status_code == 200
