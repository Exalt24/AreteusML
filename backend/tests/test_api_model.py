"""Tests for model management endpoints."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def _mock_model_state():
    with patch("backend.app.api.routes.model.model_state") as mock:
        mock.return_value = {
            "model_name": "answerdotai/ModernBERT-base",
            "model_path": "/mock/model.onnx",
            "backend": "onnx",
            "loaded_at": "2025-01-01T00:00:00",
            "status": "loaded",
        }
        yield mock


@pytest.fixture
def _mock_model_health():
    with patch("backend.app.api.routes.model.get_model") as mock:
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
        session = MagicMock()
        session.get_inputs.return_value = [MagicMock(name="input_ids"), MagicMock(name="attention_mask")]
        session.run.return_value = [[[0.1, 0.9]]]
        mock.return_value = {
            "tokenizer": tokenizer,
            "session": session,
            "backend": "onnx",
        }
        yield mock


@pytest.fixture
def _mock_reload():
    with patch("backend.app.api.routes.model.reload_model") as mock:
        mock.return_value = {"backend": "onnx"}
        yield mock


class TestModelInfo:
    def test_model_info(self, test_client, _mock_model_state):
        resp = test_client.get("/model/info")
        assert resp.status_code == 200

    def test_model_info_fields(self, test_client, _mock_model_state):
        resp = test_client.get("/model/info")
        data = resp.json()
        assert "model_name" in data
        assert "model_path" in data
        assert "backend" in data
        assert "loaded_at" in data
        assert "status" in data
        assert data["status"] == "loaded"


class TestModelHealth:
    def test_model_health(self, test_client, _mock_model_health):
        resp = test_client.get("/model/health")
        assert resp.status_code == 200

    def test_model_health_returns_prediction(self, test_client, _mock_model_health):
        resp = test_client.get("/model/health")
        data = resp.json()
        assert "healthy" in data
        assert "inference_ok" in data
        assert "message" in data


class TestModelReload:
    def test_model_reload_unauthorized(self, test_client, _mock_reload):
        resp = test_client.post("/model/reload")
        assert resp.status_code in (401, 403)

    def test_model_reload_authorized(self, test_client, auth_headers, _mock_reload):
        resp = test_client.post("/model/reload", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "reloaded"

    def test_model_reload_invalid_token(self, test_client, _mock_reload):
        resp = test_client.post("/model/reload", headers={"Authorization": "Bearer invalid.token.here"})
        assert resp.status_code == 401

    def test_model_reload_expired_token(self, test_client, expired_token_headers, _mock_reload):
        resp = test_client.post("/model/reload", headers=expired_token_headers)
        assert resp.status_code == 401
