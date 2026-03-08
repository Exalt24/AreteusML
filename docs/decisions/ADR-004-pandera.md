# ADR-004: Pandera for Data Validation

## Context

Need data validation to catch schema drift and bad inputs before they reach training or inference. Candidates evaluated: Great Expectations, Pandera, custom validation.

## Decision

Use Pandera.

## Consequences

- Lightweight and Pythonic: schemas are dataclasses, not YAML configs or notebook-driven suites
- Integrates directly with pandas DataFrames, which is already our data format throughout the pipeline
- Great Expectations is enterprise-grade overkill (data docs, expectation stores, checkpoints) for a single-dataset project
- Custom validation always starts simple and becomes unmaintainable; Pandera gives structure without overhead
