"""Tests for security middleware and protections."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def _mock_predict_for_security():
    result = MagicMock()
    result.label = 0
    result.confidence = 0.9
    result.label_name = "test_class"
    result.latency_ms = 10.0

    with (
        patch("backend.app.api.routes.predict.predict_single", new_callable=AsyncMock, return_value=result),
        patch("backend.app.api.routes.predict.get_cached_prediction", new_callable=AsyncMock, return_value=None),
        patch("backend.app.api.routes.predict.cache_prediction", new_callable=AsyncMock),
        patch("backend.app.api.routes.predict.log_prediction"),
    ):
        yield


class TestSecurityHeaders:
    def test_security_headers_present(self, test_client):
        resp = test_client.get("/health")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"
        assert resp.headers.get("X-XSS-Protection") == "1; mode=block"
        assert "Strict-Transport-Security" in resp.headers

    def test_cors_headers(self, test_client):
        resp = test_client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # CORS preflight should not error
        assert resp.status_code in (200, 405)

    def test_no_server_version_exposed(self, test_client):
        resp = test_client.get("/health")
        server_header = resp.headers.get("Server", "")
        # Should not expose specific version info
        assert "python" not in server_header.lower() or server_header == ""


class TestInputProtection:
    def test_sql_injection_prevention(self, test_client, _mock_predict_for_security):
        """SQL injection in text field should be treated as regular text, not cause errors."""
        resp = test_client.post("/predict", json={"text": "'; DROP TABLE users; --"})
        assert resp.status_code == 200

    def test_xss_prevention(self, test_client, _mock_predict_for_security):
        """Script tags in text should be treated as regular text."""
        resp = test_client.post("/predict", json={"text": "<script>alert('xss')</script>"})
        assert resp.status_code == 200

    def test_large_payload_rejection(self, test_client):
        """Payloads exceeding max_length should be rejected."""
        resp = test_client.post("/predict", json={"text": "x" * 10001})
        assert resp.status_code == 422

    def test_content_type_validation(self, test_client):
        """Non-JSON content type should be rejected."""
        resp = test_client.post(
            "/predict",
            content="text=hello",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert resp.status_code == 422


class TestRateLimiting:
    def test_rate_limiting(self, test_client):
        """Rate limiter is configured (we just verify it exists, not trigger it)."""
        from backend.app.middleware.rate_limit import limiter

        assert limiter is not None
        assert limiter._default_limits is not None
