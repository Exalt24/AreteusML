"""Prediction endpoints."""

import hashlib
import time
import uuid

from fastapi import APIRouter, BackgroundTasks, Request
from loguru import logger
from pydantic import BaseModel, Field

from backend.app.core.cache import cache_prediction, get_cached_prediction
from backend.app.middleware.rate_limit import limiter
from backend.app.services.audit import log_prediction
from backend.app.services.inference import predict_batch, predict_single
from monitoring.alerts import AlertManager
from monitoring.performance_tracker import PerformanceTracker

_tracker = PerformanceTracker()
_alert_manager = AlertManager()

router = APIRouter()

CONFIDENCE_THRESHOLD = 0.5


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to classify")


class PredictBatchRequest(BaseModel):
    texts: list[str] = Field(..., min_items=1, max_items=100, description="Texts to classify")


class PredictionResponse(BaseModel):
    label: int
    confidence: float
    label_name: str
    prediction_id: str
    latency_ms: float
    low_confidence: bool = False
    message: str | None = None


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total_latency_ms: float


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()


@router.post("", response_model=PredictionResponse)
@limiter.limit("100/minute")
async def predict(request: Request, body: PredictRequest, background_tasks: BackgroundTasks):
    """Single text classification."""
    cache_key = _text_hash(body.text)

    cached = await get_cached_prediction(cache_key)
    if cached is not None:
        logger.debug(f"Cache hit for {cache_key[:12]}...")
        cached["prediction_id"] = uuid.uuid4().hex
        return cached

    start = time.perf_counter()
    result = await predict_single(body.text)
    latency_ms = (time.perf_counter() - start) * 1000

    prediction_id = uuid.uuid4().hex

    response = PredictionResponse(
        label=result.label,
        confidence=result.confidence,
        label_name=result.label_name,
        prediction_id=prediction_id,
        latency_ms=round(latency_ms, 2),
        low_confidence=result.confidence < CONFIDENCE_THRESHOLD,
        message="Low confidence - consider human review" if result.confidence < CONFIDENCE_THRESHOLD else None,
    )

    await cache_prediction(cache_key, response.model_dump())
    background_tasks.add_task(log_prediction, body.text, response.model_dump())
    background_tasks.add_task(
        _tracker.record_prediction,
        latency_ms=round(latency_ms, 2),
        confidence=result.confidence,
        label=result.label_name,
    )
    background_tasks.add_task(
        _alert_manager.check_all_alerts,
        {"mean_confidence": result.confidence, "latency_p95": round(latency_ms, 2)},
    )

    return response


@router.post("/batch", response_model=BatchPredictionResponse)
@limiter.limit("20/minute")
async def predict_batch_endpoint(request: Request, body: PredictBatchRequest, background_tasks: BackgroundTasks):
    """Batch text classification."""
    start = time.perf_counter()
    results = await predict_batch(body.texts)
    total_latency = (time.perf_counter() - start) * 1000

    predictions = []
    for text, result in zip(body.texts, results, strict=False):
        pred = PredictionResponse(
            label=result.label,
            confidence=result.confidence,
            label_name=result.label_name,
            prediction_id=uuid.uuid4().hex,
            latency_ms=round(result.latency_ms, 2),
            low_confidence=result.confidence < CONFIDENCE_THRESHOLD,
            message="Low confidence - consider human review" if result.confidence < CONFIDENCE_THRESHOLD else None,
        )
        predictions.append(pred)
        cache_key = _text_hash(text)
        await cache_prediction(cache_key, pred.model_dump())

    background_tasks.add_task(
        lambda: [log_prediction(t, p.model_dump()) for t, p in zip(body.texts, predictions, strict=False)]
    )

    return BatchPredictionResponse(predictions=predictions, total_latency_ms=round(total_latency, 2))
