# ADR-002: Dagster as Pipeline Orchestrator

## Context

Need an ML pipeline orchestrator to manage data ingestion, training, evaluation, and export steps. Candidates evaluated: Airflow, Prefect, Dagster.

## Decision

Use Dagster.

## Consequences

- Asset-centric model maps naturally to ML artifacts (datasets, trained models, metrics) rather than task-centric DAGs
- Excellent local dev experience with built-in UI, no external scheduler or database needed
- Type-checked IO managers enforce contracts between pipeline steps, catching integration bugs early
- Simpler operational footprint than Airflow (no separate webserver, scheduler, worker processes)
