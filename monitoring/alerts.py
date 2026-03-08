"""Alert system for monitoring model health and performance."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from loguru import logger


class AlertSeverity(str, Enum):
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    DRIFT_DETECTED = "drift_detected"
    CONFIDENCE_DROP = "confidence_drop"
    ERROR_RATE_SPIKE = "error_rate_spike"
    LATENCY_SPIKE = "latency_spike"


@dataclass
class Alert:
    """A single alert event."""

    alert_type: AlertType
    severity: AlertSeverity
    message: str
    value: float
    threshold: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict:
        return {
            "type": self.alert_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
        }


class AlertManager:
    """Manages alert thresholds and checks metrics against them.

    Args:
        confidence_threshold: Minimum acceptable mean confidence.
        latency_p95_threshold: Maximum acceptable p95 latency in ms.
        error_rate_threshold: Maximum acceptable error rate (0-1).
        alert_log_path: Path to persist alerts as JSONL.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        latency_p95_threshold: float = 100.0,
        error_rate_threshold: float = 0.05,
        alert_log_path: str | Path = "monitoring/reports/alerts.jsonl",
    ):
        self.confidence_threshold = confidence_threshold
        self.latency_p95_threshold = latency_p95_threshold
        self.error_rate_threshold = error_rate_threshold
        self.alert_log_path = Path(alert_log_path)

    def check_all_alerts(self, metrics: dict) -> list[Alert]:
        """Check all alert conditions against provided metrics.

        Args:
            metrics: Dict with optional keys:
                - drift_detected (bool)
                - mean_confidence (float)
                - latency_p95 (float, ms)
                - error_rate (float, 0-1)

        Returns:
            List of triggered alerts.
        """
        alerts: list[Alert] = []

        if metrics.get("drift_detected", False):
            alerts.append(
                Alert(
                    alert_type=AlertType.DRIFT_DETECTED,
                    severity=AlertSeverity.CRITICAL,
                    message="Data drift detected in production predictions.",
                    value=1.0,
                    threshold=0.0,
                )
            )

        mean_confidence = metrics.get("mean_confidence")
        if mean_confidence is not None and mean_confidence < self.confidence_threshold:
            alerts.append(
                Alert(
                    alert_type=AlertType.CONFIDENCE_DROP,
                    severity=AlertSeverity.WARNING,
                    message=f"Mean confidence {mean_confidence:.3f} below threshold {self.confidence_threshold}.",
                    value=mean_confidence,
                    threshold=self.confidence_threshold,
                )
            )

        error_rate = metrics.get("error_rate")
        if error_rate is not None and error_rate > self.error_rate_threshold:
            alerts.append(
                Alert(
                    alert_type=AlertType.ERROR_RATE_SPIKE,
                    severity=AlertSeverity.CRITICAL,
                    message=f"Error rate {error_rate:.3%} exceeds threshold {self.error_rate_threshold:.3%}.",
                    value=error_rate,
                    threshold=self.error_rate_threshold,
                )
            )

        latency_p95 = metrics.get("latency_p95")
        if latency_p95 is not None and latency_p95 > self.latency_p95_threshold:
            alerts.append(
                Alert(
                    alert_type=AlertType.LATENCY_SPIKE,
                    severity=AlertSeverity.WARNING,
                    message=f"P95 latency {latency_p95:.1f}ms exceeds threshold {self.latency_p95_threshold:.1f}ms.",
                    value=latency_p95,
                    threshold=self.latency_p95_threshold,
                )
            )

        for alert in alerts:
            logger.warning("[{}] {}", alert.alert_type.value, alert.message)

        if alerts:
            self._save_alerts(alerts)

        return alerts

    def _save_alerts(self, alerts: list[Alert]) -> None:
        """Append alerts to the JSONL log file."""
        import json

        self.alert_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.alert_log_path, "a") as f:
            for alert in alerts:
                f.write(json.dumps(alert.to_dict()) + "\n")
        logger.info("Saved {} alert(s) to {}", len(alerts), self.alert_log_path)
