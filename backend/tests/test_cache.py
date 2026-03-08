"""Tests for Redis cache layer."""

import json
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture
def mock_redis_client():
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.setex = AsyncMock()
    return redis


class TestCacheMiss:
    @pytest.mark.asyncio
    async def test_cache_miss(self, mock_redis_client):
        mock_redis_client.get.return_value = None
        with patch("backend.app.core.cache.get_redis", new_callable=AsyncMock, return_value=mock_redis_client):
            from backend.app.core.cache import get_cached_prediction

            result = await get_cached_prediction("nonexistent_key")
            assert result is None


class TestCacheHit:
    @pytest.mark.asyncio
    async def test_cache_hit(self, mock_redis_client):
        cached = {"label": 5, "confidence": 0.9, "label_name": "test"}
        mock_redis_client.get.return_value = json.dumps(cached)
        with patch("backend.app.core.cache.get_redis", new_callable=AsyncMock, return_value=mock_redis_client):
            from backend.app.core.cache import get_cached_prediction

            result = await get_cached_prediction("some_key")
            assert result == cached


class TestCacheDifferentTexts:
    @pytest.mark.asyncio
    async def test_cache_different_texts(self, mock_redis_client):
        call_count = 0

        async def mock_get(key):
            nonlocal call_count
            call_count += 1
            if "key_a" in key:
                return json.dumps({"label": 1})
            return None

        mock_redis_client.get = mock_get
        with patch("backend.app.core.cache.get_redis", new_callable=AsyncMock, return_value=mock_redis_client):
            from backend.app.core.cache import get_cached_prediction

            result_a = await get_cached_prediction("key_a")
            result_b = await get_cached_prediction("key_b")
            assert result_a is not None
            assert result_b is None


class TestCacheExpiry:
    @pytest.mark.asyncio
    async def test_cache_expiry(self, mock_redis_client):
        """Verify cache_prediction calls setex with TTL."""
        with patch("backend.app.core.cache.get_redis", new_callable=AsyncMock, return_value=mock_redis_client):
            from backend.app.core.cache import DEFAULT_TTL, cache_prediction

            await cache_prediction("test_key", {"label": 1})
            mock_redis_client.setex.assert_called_once()
            args = mock_redis_client.setex.call_args
            assert args[0][1] == DEFAULT_TTL  # TTL argument


class TestCacheConnectionFailure:
    @pytest.mark.asyncio
    async def test_cache_connection_failure_graceful(self):
        """Cache failures should not raise - they log and return None."""
        mock_redis = AsyncMock()
        mock_redis.get.side_effect = ConnectionError("Redis unavailable")
        with patch("backend.app.core.cache.get_redis", new_callable=AsyncMock, return_value=mock_redis):
            from backend.app.core.cache import get_cached_prediction

            result = await get_cached_prediction("any_key")
            assert result is None
