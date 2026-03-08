"""Dagster Definitions - combines training and monitoring pipelines."""

from dagster import Definitions

from ml.pipelines.monitoring_pipeline import (
    MonitoringConfig,
    alert_check,
    current_predictions,
    drift_report,
    monitoring_job,
    monitoring_sensor,
    reference_data,
    retrain_trigger,
)
from ml.pipelines.training_pipeline import (
    MLflowResource,
    ModelPathResource,
    augmented_data,
    evaluated_model,
    existing_model,
    raw_data,
    registered_model,
    trained_model,
    validated_data,
)

defs = Definitions(
    assets=[
        # Training pipeline
        raw_data,
        validated_data,
        augmented_data,
        trained_model,
        existing_model,
        evaluated_model,
        registered_model,
        # Monitoring pipeline
        reference_data,
        current_predictions,
        drift_report,
        alert_check,
        retrain_trigger,
    ],
    jobs=[monitoring_job],
    sensors=[monitoring_sensor],
    resources={
        "mlflow_resource": MLflowResource(
            tracking_uri="file:./mlruns",
            experiment_name="banking77-modernbert",
        ),
        "model_paths": ModelPathResource(
            base_dir="artifacts/models",
            data_dir="artifacts/data",
        ),
        "monitoring_config": MonitoringConfig(
            drift_threshold=0.15,
            confidence_drop_threshold=0.05,
            min_samples=50,
            reference_data_path="artifacts/data/reference_predictions.parquet",
            current_data_path="artifacts/data/current_predictions.parquet",
        ),
    },
)
