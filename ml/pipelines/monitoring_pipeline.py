"""Dagster monitoring pipeline assets for drift detection and retraining triggers."""

import pandas as pd
from dagster import (
    AssetExecutionContext,
    ConfigurableResource,
    DefaultSensorStatus,
    RunRequest,
    SensorEvaluationContext,
    asset,
    define_asset_job,
    sensor,
)
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from loguru import logger

# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


class MonitoringConfig(ConfigurableResource):
    drift_threshold: float = 0.15
    confidence_drop_threshold: float = 0.05
    min_samples: int = 50
    reference_data_path: str = "artifacts/data/reference_predictions.parquet"
    current_data_path: str = "artifacts/data/current_predictions.parquet"


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------


@asset(group_name="monitoring")
def reference_data(
    context: AssetExecutionContext,
    monitoring_config: MonitoringConfig,
) -> pd.DataFrame:
    """Load test-set predictions from training time as the reference distribution."""
    path = monitoring_config.reference_data_path
    logger.info(f"Loading reference data from {path}")

    df = pd.read_parquet(path)
    context.log.info(f"Reference data loaded: {len(df)} rows, columns: {list(df.columns)}")
    return df


@asset(group_name="monitoring")
def current_predictions(
    context: AssetExecutionContext,
    monitoring_config: MonitoringConfig,
) -> pd.DataFrame:
    """Load recent predictions from local parquet file."""
    path = monitoring_config.current_data_path
    logger.info(f"Loading current predictions from {path}")

    df = pd.read_parquet(path)

    logger.info(f"Loaded {len(df)} recent predictions from {path}")
    context.log.info(f"Current predictions: {len(df)} rows")

    if len(df) < monitoring_config.min_samples:
        logger.warning(
            f"Only {len(df)} samples (min: {monitoring_config.min_samples}). Drift detection may be unreliable."
        )

    return df


@asset(group_name="monitoring")
def drift_report(
    context: AssetExecutionContext,
    reference_data: pd.DataFrame,
    current_predictions: pd.DataFrame,
) -> dict:
    """Run Evidently drift detection between reference and current data."""
    # Align columns for comparison
    shared_cols = list(set(reference_data.columns) & set(current_predictions.columns))
    shared_cols = [c for c in shared_cols if c not in ("created_at",)]

    if not shared_cols:
        logger.error("No shared columns between reference and current data")
        return {"error": "no_shared_columns", "drift_detected": False}

    ref = reference_data[shared_cols].copy()
    cur = current_predictions[shared_cols].copy()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)

    result = report.as_dict()

    drift_summary = {
        "dataset_drift": result["metrics"][0]["result"].get("dataset_drift", False),
        "drift_share": result["metrics"][0]["result"].get("drift_share", 0.0),
        "number_of_columns": result["metrics"][0]["result"].get("number_of_columns", 0),
        "number_of_drifted_columns": result["metrics"][0]["result"].get("number_of_drifted_columns", 0),
    }

    logger.info(f"Drift report: {drift_summary}")
    context.log.info(f"Dataset drift detected: {drift_summary['dataset_drift']}")
    return drift_summary


@asset(group_name="monitoring")
def alert_check(
    context: AssetExecutionContext,
    drift_report: dict,
    current_predictions: pd.DataFrame,
    reference_data: pd.DataFrame,
    monitoring_config: MonitoringConfig,
) -> dict:
    """Check all alert thresholds and produce alert summary."""
    alerts: list[str] = []

    # 1. Data drift alert
    drift_share = drift_report.get("drift_share", 0.0)
    if drift_share > monitoring_config.drift_threshold:
        alerts.append(
            f"DATA_DRIFT: {drift_share:.2%} columns drifted (threshold: {monitoring_config.drift_threshold:.2%})"
        )

    # 2. Confidence drop alert
    if "confidence" in current_predictions.columns and "confidence" in reference_data.columns:
        ref_confidence = reference_data["confidence"].mean()
        cur_confidence = current_predictions["confidence"].mean()
        drop = ref_confidence - cur_confidence

        if drop > monitoring_config.confidence_drop_threshold:
            alerts.append(
                f"CONFIDENCE_DROP: {drop:.4f} "
                f"(ref={ref_confidence:.4f}, cur={cur_confidence:.4f}, "
                f"threshold={monitoring_config.confidence_drop_threshold})"
            )

    # 3. Label distribution shift
    if "predicted_label" in current_predictions.columns and "predicted_label" in reference_data.columns:
        ref_dist = reference_data["predicted_label"].value_counts(normalize=True)
        cur_dist = current_predictions["predicted_label"].value_counts(normalize=True)
        all_labels = set(ref_dist.index) | set(cur_dist.index)
        max_shift = max(abs(ref_dist.get(label, 0) - cur_dist.get(label, 0)) for label in all_labels)
        if max_shift > 0.10:
            alerts.append(f"LABEL_SHIFT: max single-label shift {max_shift:.2%}")

    alert_summary = {
        "alerts": alerts,
        "alert_count": len(alerts),
        "needs_attention": len(alerts) > 0,
    }

    if alerts:
        for a in alerts:
            logger.warning(f"ALERT: {a}")
    else:
        logger.info("No alerts triggered")

    context.log.info(f"Alert check: {len(alerts)} alert(s)")
    return alert_summary


@asset(group_name="monitoring")
def retrain_trigger(
    context: AssetExecutionContext,
    drift_report: dict,
    alert_check: dict,
) -> dict:
    """Determine if retraining is needed based on drift and alert signals."""
    drift_detected = drift_report.get("dataset_drift", False)
    has_alerts = alert_check.get("needs_attention", False)

    should_retrain = drift_detected or has_alerts

    reasons: list[str] = []
    if drift_detected:
        reasons.append("dataset_drift_detected")
    if has_alerts:
        reasons.extend(alert_check.get("alerts", []))

    result = {
        "should_retrain": should_retrain,
        "reasons": reasons,
        "drift_detected": drift_detected,
        "alert_count": alert_check.get("alert_count", 0),
    }

    if should_retrain:
        logger.warning(f"RETRAIN RECOMMENDED: {reasons}")
    else:
        logger.info("No retraining needed at this time")

    context.log.info(f"Retrain trigger: {should_retrain} (reasons: {len(reasons)})")
    return result


# ---------------------------------------------------------------------------
# Sensor - run monitoring every 5 minutes
# ---------------------------------------------------------------------------

monitoring_job = define_asset_job(
    name="monitoring_job",
    selection=[
        reference_data,
        current_predictions,
        drift_report,
        alert_check,
        retrain_trigger,
    ],
)


@sensor(
    job=monitoring_job,
    minimum_interval_seconds=300,
    default_status=DefaultSensorStatus.RUNNING,
)
def monitoring_sensor(context: SensorEvaluationContext):
    """Trigger monitoring pipeline every 5 minutes."""
    yield RunRequest(run_key=None)
