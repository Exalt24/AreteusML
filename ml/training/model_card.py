"""Generate a model card with metrics, bias considerations, and EU AI Act awareness.

Run with:
    python -m ml.training.model_card
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from textwrap import dedent

import mlflow
from loguru import logger
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = PROJECT_ROOT / "ml" / "models" / "production"

console = Console()

# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------
MODEL_CARD_TEMPLATE = dedent("""\
    # Model Card: AreteusML Banking Intent Classifier

    **Generated:** {generated_date}

    ## Model Description

    Fine-tuned **ModernBERT-base** (`answerdotai/ModernBERT-base`) for intent
    classification on the Banking77 dataset. The model classifies customer
    banking queries into 77 intent categories to support automated routing
    and customer service workflows.

    - **Architecture:** ModernBERT-base (encoder-only transformer)
    - **Task:** Multi-class text classification (77 classes)
    - **Training data:** Banking77 (13,083 training samples)
    - **Language:** English
    - **Fine-tuning:** Class-weighted cross-entropy loss with early stopping

    ## Intended Use

    - **Primary:** Research and educational demonstration of production ML
      pipeline practices (data validation, experiment tracking, model export,
      explainability).
    - **Secondary:** Prototyping automated intent routing for banking chatbots.
    - **Out of scope:** Direct deployment in production financial systems
      without additional validation, multilingual support, or handling of
      adversarial/out-of-distribution inputs.

    ## Performance Metrics

    ### ModernBERT (Fine-tuned)
    {modernbert_metrics}

    ### Best Classical Baseline (TF-IDF + LogReg)
    {baseline_metrics}

    ### ONNX Export Benchmarks
    {onnx_benchmarks}

    ## Training Details

    - **Optimizer:** AdamW
    - **Learning rate:** Tuned via Optuna hyperparameter search
    - **Epochs:** Up to 10 with early stopping (patience=2)
    - **Class weighting:** Inverse frequency weights to handle class imbalance
    - **Evaluation:** Stratified train/test split
    - **Experiment tracking:** MLflow
    - **Seed:** 42 (full reproducibility across runs)

    ## Limitations

    - **Domain specificity:** Trained exclusively on banking queries; will not
      generalize to other domains without fine-tuning.
    - **Language:** English only. No multilingual capability.
    - **Class granularity:** Some Banking77 classes are semantically overlapping
      (e.g., "card_arrival" vs "card_delivery_estimate"), which creates an
      inherent ceiling on achievable accuracy.
    - **Input length:** Truncated to 128 tokens; longer queries may lose context.
    - **Temporal drift:** Training data is static; real-world banking language
      evolves and may degrade performance over time.

    ## Bias Considerations

    - **Dataset bias:** Banking77 is crowd-sourced and may not represent the
      full diversity of real banking customers (accents, dialects, phrasing
      patterns across demographics).
    - **Class imbalance:** Some intents are underrepresented. Class weighting
      mitigates but does not eliminate this issue.
    - **Fairness:** No explicit fairness auditing has been performed across
      protected attributes (age, gender, ethnicity). Any production deployment
      should include bias testing with representative user populations.
    - **Geographic bias:** Training data skews toward English-speaking banking
      terminology and may not reflect banking practices in all regions.

    ## Ethical Considerations

    - This model processes financial-domain text. Misclassification could lead
      to incorrect routing of sensitive customer requests (e.g., fraud reports
      being routed to general inquiries).
    - The model should always be paired with a human fallback mechanism in any
      customer-facing deployment.

    ## EU AI Act Risk Classification Awareness

    Under the EU AI Act framework:

    - **Risk category:** This model would likely fall under **limited risk** if
      used for internal routing assistance, or **high risk** if used as part of
      a system that makes or influences decisions about financial services
      access.
    - **Transparency:** Users interacting with a system powered by this model
      should be informed they are interacting with an AI system.
    - **Human oversight:** A human-in-the-loop is recommended for any
      consequential decisions based on model predictions.
    - **Documentation:** This model card serves as part of the technical
      documentation requirement.
    - **Data governance:** Training data (Banking77) is publicly available and
      does not contain personal data.

    ## How to Use

    ```python
    from transformers import pipeline

    clf = pipeline("text-classification", model="ml/models/production")
    result = clf("I need to activate my new card")
    print(result)
    ```

    ## Citation

    ```
    @inproceedings{{casanueva2020efficient,
        title={{Efficient Intent Detection with Dual Sentence Encoders}},
        author={{Casanueva et al.}},
        booktitle={{NLP for ConvAI - ACL 2020}},
        year={{2020}}
    }}
    ```
""")


def _format_metrics(metrics: dict, prefix: str = "") -> str:
    """Format a metrics dict as a markdown table."""
    if not metrics:
        return "_No metrics available._"

    lines = ["| Metric | Value |", "|--------|-------|"]
    for key, value in sorted(metrics.items()):
        display_key = key.replace(prefix, "").replace("_", " ").title()
        if isinstance(value, float):
            lines.append(f"| {display_key} | {value:.4f} |")
        else:
            lines.append(f"| {display_key} | {value} |")
    return "\n".join(lines)


def load_mlflow_metrics(experiment_name: str, run_name: str | None = None) -> dict:
    """Load metrics from the best MLflow run in an experiment."""
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.warning(f"MLflow experiment '{experiment_name}' not found")
            return {}

        filter_str = f"tags.mlflow.runName = '{run_name}'" if run_name else ""
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=filter_str,
            order_by=["metrics.f1_macro DESC"],
            max_results=1,
        )
        if not runs:
            return {}

        return runs[0].data.metrics
    except Exception as e:
        logger.warning(f"Could not load MLflow metrics: {e}")
        return {}


def load_onnx_benchmarks() -> str:
    """Load ONNX benchmark results if available."""
    benchmark_path = PROJECT_ROOT / "ml" / "models" / "onnx" / "benchmark_results.json"
    if not benchmark_path.exists():
        return "_ONNX benchmarks not yet generated. Run `python -m ml.training.export_onnx`._"

    with open(benchmark_path) as f:
        results = json.load(f)

    lines = [
        "| Variant | Latency (ms/sample) | Speedup |",
        "|---------|--------------------:|--------:|",
        f"| PyTorch FP32 | {results['pytorch_fp32_ms']:.2f} | 1.00x |",
        f"| ONNX FP32 | {results['onnx_fp32_ms']:.2f} | {results['speedup_onnx_fp32']:.2f}x |",
        f"| ONNX INT8 | {results['onnx_int8_ms']:.2f} | {results['speedup_onnx_int8']:.2f}x |",
    ]
    return "\n".join(lines)


def main() -> None:
    """Generate the model card markdown file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    modernbert_metrics = load_mlflow_metrics("Banking77-ModernBERT")
    baseline_metrics = load_mlflow_metrics("Banking77-Baselines", "TF-IDF + LogReg")
    onnx_benchmarks = load_onnx_benchmarks()

    card = MODEL_CARD_TEMPLATE.format(
        generated_date=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        modernbert_metrics=_format_metrics(modernbert_metrics),
        baseline_metrics=_format_metrics(baseline_metrics),
        onnx_benchmarks=onnx_benchmarks,
    )

    output_path = OUTPUT_DIR / "MODEL_CARD.md"
    output_path.write_text(card, encoding="utf-8")

    console.print(f"\n[green]Model card saved to {output_path}[/green]")
    logger.success(f"Model card generated at {output_path}")


if __name__ == "__main__":
    main()
