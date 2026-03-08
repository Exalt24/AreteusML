"""Track inference performance metrics with SQLite (history) and in-memory buffer (real-time)."""

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger
from sqlalchemy import Column, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session

# Time windows in seconds
WINDOWS = {
    "1min": 60,
    "5min": 300,
    "1hour": 3600,
}

# --- SQLAlchemy models ---

class Base(DeclarativeBase):
    pass


class PredictionMetric(Base):
    __tablename__ = "prediction_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(Float, nullable=False, index=True)
    latency_ms = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    label = Column(String(255), nullable=True)


# --- Engine singleton ---

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        root = Path(__file__).resolve().parents[1]
        db_path = root / "data" / "monitoring.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(_engine)
        logger.info("Monitoring DB initialized at {}", db_path)
    return _engine


# --- Tracker ---

@dataclass
class PerformanceTracker:
    """Track and aggregate inference performance metrics.

    Uses an in-memory buffer for real-time windowed metrics and SQLite
    for persistent historical storage.
    """

    _buffer: list = field(default_factory=list, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _trim_buffer(self, now: float) -> None:
        """Remove entries older than 1 hour. Must be called with lock held."""
        cutoff = now - WINDOWS["1hour"]
        self._buffer = [e for e in self._buffer if e[0] >= cutoff]

    def record_prediction(
        self,
        latency_ms: float,
        confidence: float,
        label: str | None = None,
    ) -> None:
        """Record a single prediction's metrics.

        Args:
            latency_ms: Inference latency in milliseconds.
            confidence: Model confidence score (0-1).
            label: Predicted label/class.
        """
        ts = time.time()

        with self._lock:
            self._buffer.append((ts, latency_ms, confidence, label))
            self._trim_buffer(ts)

        # Persist to SQLite
        try:
            engine = _get_engine()
            with Session(engine) as session:
                session.add(PredictionMetric(
                    timestamp=ts,
                    latency_ms=latency_ms,
                    confidence=confidence,
                    label=label,
                ))
                session.commit()
        except Exception:
            logger.exception("Failed to write to SQLite, metrics stored in memory only")

        logger.debug(
            "Recorded prediction: latency={:.1f}ms confidence={:.3f} label={}",
            latency_ms,
            confidence,
            label,
        )

    def get_metrics_summary(self, window: str = "5min") -> dict:
        """Aggregate metrics over a time window from in-memory buffer.

        Args:
            window: One of '1min', '5min', '1hour'.

        Returns:
            Dict with keys: count, latency_mean, latency_p50, latency_p95,
            confidence_mean, confidence_min, throughput_per_sec.
        """
        if window not in WINDOWS:
            raise ValueError(f"Invalid window '{window}'. Choose from {list(WINDOWS.keys())}")

        now = time.time()
        cutoff = now - WINDOWS[window]

        with self._lock:
            entries = [(ts, lat, conf) for ts, lat, conf, _lbl in self._buffer if ts >= cutoff]

        if not entries:
            logger.info("No predictions in the last {} window.", window)
            return {
                "window": window,
                "count": 0,
                "latency_mean": 0.0,
                "latency_p50": 0.0,
                "latency_p95": 0.0,
                "confidence_mean": 0.0,
                "confidence_min": 0.0,
                "throughput_per_sec": 0.0,
            }

        latencies = np.array([e[1] for e in entries])
        confidences = np.array([e[2] for e in entries])
        count = len(entries)
        elapsed = WINDOWS[window]

        summary = {
            "window": window,
            "count": count,
            "latency_mean": float(np.mean(latencies)),
            "latency_p50": float(np.percentile(latencies, 50)),
            "latency_p95": float(np.percentile(latencies, 95)),
            "confidence_mean": float(np.mean(confidences)),
            "confidence_min": float(np.min(confidences)),
            "throughput_per_sec": count / elapsed,
        }

        logger.info(
            "Metrics [{}]: {} predictions, latency p95={:.1f}ms, confidence mean={:.3f}",
            window,
            count,
            summary["latency_p95"],
            summary["confidence_mean"],
        )

        return summary

    def get_metrics(self) -> dict:
        """Backward-compatible shorthand for get_metrics_summary('5min')."""
        return self.get_metrics_summary("5min")

    def close(self) -> None:
        """Clean up resources."""
        with self._lock:
            self._buffer.clear()
        logger.info("Performance tracker closed.")
