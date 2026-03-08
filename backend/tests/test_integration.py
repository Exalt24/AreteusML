"""Integration tests for the AreteusML API.

These tests run against a live server instance.
Run with: pytest -m integration backend/tests/test_integration.py

Requires the API to be running at API_URL (default: http://localhost:8000).
"""

import os

import httpx
import pytest

BASE_URL = os.getenv("API_URL", "http://localhost:8000")

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as c:
        yield c


def test_health(client: httpx.Client):
    response = client.get("/health")
    assert response.status_code == 200


def test_predict(client: httpx.Client):
    response = client.post("/predict", json={"text": "I want to check my balance"})
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "confidence" in data
    assert "label_name" in data
    assert "latency_ms" in data
    assert "prediction_id" in data


def test_predict_batch(client: httpx.Client):
    response = client.post(
        "/predict/batch",
        json={"texts": ["I want to check my balance", "How do I activate my card?"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2


def test_feedback(client: httpx.Client):
    # First make a prediction to get a prediction_id
    pred_response = client.post(
        "/predict", json={"text": "I want to check my balance"}
    )
    assert pred_response.status_code == 200
    prediction_id = pred_response.json()["prediction_id"]

    # Submit feedback for that prediction
    response = client.post(
        "/feedback",
        json={
            "prediction_id": prediction_id,
            "correct_label": "balance_not_updated_after_bank_transfer",
            "is_correct": False,
        },
    )
    assert response.status_code == 200


def test_feedback_stats(client: httpx.Client):
    response = client.get("/feedback/stats")
    assert response.status_code == 200
