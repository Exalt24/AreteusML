"""Audit logging service with SQLAlchemy models."""

import json
import uuid
from datetime import UTC, datetime

from loguru import logger
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session

from backend.app.core.config import settings


class Base(DeclarativeBase):
    pass


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    action = Column(String(50), nullable=False, index=True)
    details = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


class FeedbackLog(Base):
    __tablename__ = "feedback_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    feedback_id = Column(String(64), unique=True, nullable=False)
    prediction_id = Column(String(64), nullable=False, index=True)
    correct_label = Column(Integer, nullable=False)
    correct_label_name = Column(String(100), nullable=True)
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        try:
            _engine = create_engine(settings.database_url, pool_pre_ping=True)
            # Test connection
            with _engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            Base.metadata.create_all(_engine)
        except Exception as e:
            logger.warning(f"PostgreSQL not available ({e}), using SQLite for audit logs")
            from pathlib import Path

            root = Path(__file__).resolve().parents[3]
            sqlite_path = root / "data" / "audit.db"
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            _engine = create_engine(f"sqlite:///{sqlite_path}")
            Base.metadata.create_all(_engine)
    return _engine


def log_prediction(text: str, prediction: dict):
    """Log a prediction to the audit table."""
    try:
        engine = _get_engine()
        with Session(engine) as session:
            entry = AuditLog(
                action="prediction",
                details=json.dumps({"text_preview": text[:200], "prediction": prediction}),
            )
            session.add(entry)
            session.commit()
    except Exception as e:
        logger.warning(f"Failed to log prediction: {e}")


async def log_feedback(
    prediction_id: str,
    correct_label: int,
    correct_label_name: str | None = None,
    comment: str | None = None,
) -> str:
    """Log feedback and return feedback_id."""
    feedback_id = uuid.uuid4().hex
    try:
        engine = _get_engine()
        with Session(engine) as session:
            fb = FeedbackLog(
                feedback_id=feedback_id,
                prediction_id=prediction_id,
                correct_label=correct_label,
                correct_label_name=correct_label_name,
                comment=comment,
            )
            session.add(fb)

            audit = AuditLog(
                action="feedback",
                details=json.dumps(
                    {
                        "feedback_id": feedback_id,
                        "prediction_id": prediction_id,
                        "correct_label": correct_label,
                    }
                ),
            )
            session.add(audit)
            session.commit()
    except Exception as e:
        logger.warning(f"Failed to log feedback: {e}")
    return feedback_id


async def get_feedback_stats() -> dict:
    """Get feedback statistics."""
    try:
        engine = _get_engine()
        with Session(engine) as session:
            from sqlalchemy import func

            total = session.query(func.count(FeedbackLog.id)).scalar() or 0

            top_classes = (
                session.query(
                    FeedbackLog.correct_label,
                    FeedbackLog.correct_label_name,
                    func.count(FeedbackLog.id).label("count"),
                )
                .group_by(FeedbackLog.correct_label, FeedbackLog.correct_label_name)
                .order_by(func.count(FeedbackLog.id).desc())
                .limit(10)
                .all()
            )

            return {
                "total_corrections": total,
                "top_corrected_classes": [
                    {"label": r[0], "label_name": r[1] or f"class_{r[0]}", "count": r[2]} for r in top_classes
                ],
            }
    except Exception as e:
        logger.warning(f"Failed to get feedback stats: {e}")
        return {"total_corrections": 0, "top_corrected_classes": []}


async def get_recent_audit_logs(limit: int = 50) -> list[dict]:
    """Get recent audit log entries."""
    try:
        engine = _get_engine()
        with Session(engine) as session:
            entries = session.query(AuditLog).order_by(AuditLog.id.desc()).limit(limit).all()
            return [
                {
                    "id": e.id,
                    "action": e.action,
                    "details": e.details,
                    "created_at": e.created_at.isoformat() if e.created_at else "",
                }
                for e in entries
            ]
    except Exception as e:
        logger.warning(f"Failed to get audit logs: {e}")
        return []
