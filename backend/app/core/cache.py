"""Redis cache for prediction results."""

import json

from loguru import logger
from redis.asyncio import Redis

from backend.app.core.config import settings

_redis: Redis | None = None

DEFAULT_TTL = 3600  # 1 hour


async def get_redis() -> Redis:
    """Get or create the Redis connection."""
    global _redis
    if _redis is None:
        _redis = Redis.from_url(settings.redis_url, decode_responses=True)
    return _redis


async def close_redis():
    """Close the Redis connection."""
    global _redis
    if _redis is not None:
        await _redis.close()
        _redis = None
        logger.info("Redis connection closed")


async def cache_prediction(key: str, data: dict, ttl: int = DEFAULT_TTL):
    """Cache a prediction result."""
    try:
        redis = await get_redis()
        await redis.setex(f"pred:{key}", ttl, json.dumps(data))
    except Exception as e:
        logger.warning(f"Cache write failed: {e}")


async def get_cached_prediction(key: str) -> dict | None:
    """Retrieve a cached prediction."""
    try:
        redis = await get_redis()
        raw = await redis.get(f"pred:{key}")
        if raw:
            return json.loads(raw)
    except Exception as e:
        logger.warning(f"Cache read failed: {e}")
    return None
