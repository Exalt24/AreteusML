"""AreteusML FastAPI application entry point."""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from backend.app.api.routes import admin, feedback, model, predict
from backend.app.core.cache import close_redis, get_redis
from backend.app.core.model_loader import cleanup_model, get_model
from backend.app.middleware.rate_limit import limiter
from backend.app.middleware.security_headers import SecurityHeadersMiddleware

REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["method", "endpoint"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    logger.info("Starting AreteusML API...")
    try:
        get_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.warning(f"Model not available at startup: {e}")

    try:
        redis = await get_redis()
        await redis.ping()
        logger.info("Redis connected successfully")
    except Exception as e:
        logger.warning(f"Redis not available at startup: {e}")

    yield

    logger.info("Shutting down AreteusML API...")
    cleanup_model()
    await close_redis()
    logger.info("Shutdown complete")


app = FastAPI(
    title="AreteusML API",
    version="0.1.0",
    lifespan=lifespan,
)

# Rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security headers
app.add_middleware(SecurityHeadersMiddleware)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Track request count and latency for Prometheus."""
    start = time.perf_counter()
    response: Response = await call_next(request)
    duration = time.perf_counter() - start
    endpoint = request.url.path
    REQUEST_COUNT.labels(method=request.method, endpoint=endpoint, status=response.status_code).inc()
    REQUEST_LATENCY.labels(method=request.method, endpoint=endpoint).observe(duration)
    return response


# Routes
app.include_router(predict.router, prefix="/predict", tags=["Predictions"])
app.include_router(model.router, prefix="/model", tags=["Model"])
app.include_router(feedback.router, prefix="/feedback", tags=["Feedback"])
app.include_router(admin.router, prefix="/admin", tags=["Admin"])


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
