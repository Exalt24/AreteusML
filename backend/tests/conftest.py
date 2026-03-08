"""Shared fixtures for backend tests."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def sample_text():
    return "I need to activate my new card"


@pytest.fixture
def sample_texts():
    return [
        "I need to activate my new card",
        "How do I check my balance?",
        "I want to transfer money to another account",
    ]


@pytest.fixture
def mock_prediction_result():
    """A mock PredictionResult dataclass instance."""
    from unittest.mock import MagicMock

    result = MagicMock()
    result.label = 5
    result.confidence = 0.92
    result.label_name = "activate_my_card"
    result.latency_ms = 12.5
    return result


@pytest.fixture
def mock_low_confidence_result():
    """A mock PredictionResult with low confidence."""
    result = MagicMock()
    result.label = 10
    result.confidence = 0.3
    result.label_name = "check_balance"
    result.latency_ms = 15.0
    return result


@pytest.fixture
def mock_model():
    """Mock model dict matching get_model() return shape."""
    model = MagicMock()
    tokenizer = MagicMock()
    return {
        "model": model,
        "tokenizer": tokenizer,
        "session": MagicMock(),
        "backend": "onnx",
        "label_map": {"0": "activate_my_card", "5": "check_balance"},
        "model_path": "/mock/path/model.onnx",
    }


@pytest.fixture
def mock_redis():
    """Mock async Redis client."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.setex = AsyncMock()
    redis.ping = AsyncMock()
    redis.close = AsyncMock()
    return redis


@pytest.fixture
def test_client():
    """FastAPI TestClient with all external dependencies mocked."""
    with (
        patch("backend.app.core.model_loader.get_model") as mock_get_model,
        patch("backend.app.core.model_loader.cleanup_model"),
        patch("backend.app.core.cache.get_redis") as mock_get_redis,
        patch("backend.app.core.cache.close_redis"),
        patch("backend.app.services.inference.get_model") as mock_inf_model,
    ):
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()
        mock_redis.close = AsyncMock()
        mock_get_redis.return_value = mock_redis

        mock_model_data = {
            "model": MagicMock(),
            "tokenizer": MagicMock(),
            "session": MagicMock(),
            "backend": "onnx",
            "label_map": {"0": "activate_my_card"},
            "model_path": "/mock/model.onnx",
        }
        mock_get_model.return_value = mock_model_data
        mock_inf_model.return_value = mock_model_data

        from fastapi.testclient import TestClient

        from backend.app.main import app

        with TestClient(app) as client:
            yield client


@pytest.fixture
def auth_headers():
    """Valid JWT auth headers for admin endpoints."""
    from backend.app.core.security import create_access_token

    token = create_access_token(data={"sub": "admin", "role": "admin"})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def expired_token_headers():
    """Expired JWT auth headers."""
    from datetime import timedelta

    from backend.app.core.security import create_access_token

    token = create_access_token(
        data={"sub": "admin", "role": "admin"},
        expires_delta=timedelta(seconds=-1),
    )
    return {"Authorization": f"Bearer {token}"}
