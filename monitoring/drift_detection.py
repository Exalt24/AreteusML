"""Drift detection using Evidently AI for monitoring model and data drift."""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, TextOverviewPreset
from evidently.report import Report
from loguru import logger


@dataclass
class DriftResult:
    """Result of a drift detection run."""

    drift_detected: bool
    share_of_drifted_columns: float
    drifted_columns: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    text_columns: list[str] | None = None,
    column_mapping: ColumnMapping | None = None,
) -> DriftResult:
    """Compare current production data against reference data for drift.

    Args:
        reference_df: Reference dataset (test set predictions from training time).
        current_df: Current production predictions.
        text_columns: Optional list of text column names for TextOverviewPreset.
        column_mapping: Optional Evidently column mapping.

    Returns:
        DriftResult with drift status and summary metrics.
    """
    logger.info(
        "Running drift detection: reference={} rows, current={} rows",
        len(reference_df),
        len(current_df),
    )

    metrics = [DataDriftPreset()]
    if text_columns:
        for col in text_columns:
            metrics.append(TextOverviewPreset(column_name=col))
        logger.info("Including text drift analysis for columns: {}", text_columns)

    report = Report(metrics=metrics)
    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )

    result_dict = report.as_dict()

    # Extract data drift results from the report
    data_drift = result_dict.get("metrics", [{}])[0].get("result", {})
    drift_detected = data_drift.get("dataset_drift", False)
    share_drifted = data_drift.get("share_of_drifted_columns", 0.0)
    drifted_columns = [
        col for col, info in data_drift.get("drift_by_columns", {}).items() if info.get("drift_detected", False)
    ]

    summary = {
        "number_of_columns": data_drift.get("number_of_columns", 0),
        "number_of_drifted_columns": data_drift.get("number_of_drifted_columns", 0),
        "share_of_drifted_columns": share_drifted,
    }

    if drift_detected:
        logger.warning(
            "Drift detected! {}/{} columns drifted: {}",
            summary["number_of_drifted_columns"],
            summary["number_of_columns"],
            drifted_columns,
        )
    else:
        logger.info("No significant drift detected.")

    return DriftResult(
        drift_detected=drift_detected,
        share_of_drifted_columns=share_drifted,
        drifted_columns=drifted_columns,
        summary=summary,
    )


def generate_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: str | Path = "monitoring/reports/drift_report.html",
    text_columns: list[str] | None = None,
    column_mapping: ColumnMapping | None = None,
) -> Path:
    """Generate an HTML drift report and save to disk.

    Args:
        reference_df: Reference dataset.
        current_df: Current production predictions.
        output_path: Path to save the HTML report.
        text_columns: Optional text columns for TextOverviewPreset.
        column_mapping: Optional Evidently column mapping.

    Returns:
        Path to the saved report.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Generating drift report to {}", output_path)

    metrics = [DataDriftPreset()]
    if text_columns:
        for col in text_columns:
            metrics.append(TextOverviewPreset(column_name=col))

    report = Report(metrics=metrics)
    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )
    report.save_html(str(output_path))

    logger.info("Drift report saved to {}", output_path)
    return output_path
