"""Tests for authentication and admin endpoints."""

from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture
def _mock_audit():
    with patch("backend.app.api.routes.admin.get_recent_audit_logs", new_callable=AsyncMock) as mock:
        mock.return_value = [
            {"id": 1, "action": "prediction", "details": "{}", "created_at": "2025-01-01T00:00:00"},
            {"id": 2, "action": "feedback", "details": "{}", "created_at": "2025-01-01T00:01:00"},
        ]
        yield mock


class TestTokenCreation:
    def test_create_token(self, test_client):
        resp = test_client.post(
            "/admin/token",
            json={
                "username": "admin",
                "password": "change-me-in-production",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_create_token_invalid_credentials(self, test_client):
        resp = test_client.post(
            "/admin/token",
            json={
                "username": "admin",
                "password": "wrong_password",
            },
        )
        assert resp.status_code == 401

    def test_token_format(self, test_client):
        resp = test_client.post(
            "/admin/token",
            json={
                "username": "admin",
                "password": "change-me-in-production",
            },
        )
        token = resp.json()["access_token"]
        # JWT tokens have 3 parts separated by dots
        parts = token.split(".")
        assert len(parts) == 3

    def test_token_expiry(self, test_client):
        """Verify token contains exp claim."""
        from jose import jwt

        from backend.app.core.config import settings

        resp = test_client.post(
            "/admin/token",
            json={
                "username": "admin",
                "password": "change-me-in-production",
            },
        )
        token = resp.json()["access_token"]
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        assert "exp" in payload


class TestProtectedEndpoints:
    def test_protected_endpoint_no_token(self, test_client, _mock_audit):
        resp = test_client.get("/admin/audit")
        assert resp.status_code in (401, 403)

    def test_protected_endpoint_invalid_token(self, test_client, _mock_audit):
        resp = test_client.get("/admin/audit", headers={"Authorization": "Bearer fake.token.here"})
        assert resp.status_code == 401

    def test_protected_endpoint_valid_token(self, test_client, auth_headers, _mock_audit):
        resp = test_client.get("/admin/audit", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_audit_log_requires_auth(self, test_client, _mock_audit):
        resp = test_client.get("/admin/audit")
        assert resp.status_code in (401, 403)
