"""Feedback endpoints for human-in-the-loop corrections."""

from fastapi import APIRouter, Request
from loguru import logger
from pydantic import BaseModel, Field

from backend.app.middleware.rate_limit import limiter
from backend.app.services.audit import get_feedback_stats, log_feedback

router = APIRouter()


class FeedbackRequest(BaseModel):
    prediction_id: str = Field(..., description="ID of the prediction to correct")
    correct_label: int = Field(..., ge=0, description="The correct label")
    correct_label_name: str | None = Field(None, description="Human-readable label name")
    comment: str | None = Field(None, max_length=1000)


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: str


class FeedbackStats(BaseModel):
    total_corrections: int
    top_corrected_classes: list[dict]


@router.post("", response_model=FeedbackResponse)
@limiter.limit("50/minute")
async def submit_feedback(request: Request, body: FeedbackRequest):
    """Submit a correction for a prediction."""
    logger.info(f"Feedback received for prediction {body.prediction_id}: label={body.correct_label}")
    feedback_id = await log_feedback(
        prediction_id=body.prediction_id,
        correct_label=body.correct_label,
        correct_label_name=body.correct_label_name,
        comment=body.comment,
    )
    return FeedbackResponse(status="recorded", feedback_id=feedback_id)


@router.get("/stats", response_model=FeedbackStats)
async def feedback_stats():
    """Get feedback statistics."""
    stats = await get_feedback_stats()
    return FeedbackStats(
        total_corrections=stats["total_corrections"],
        top_corrected_classes=stats["top_corrected_classes"],
    )
