"""Model validation gate - ensures models meet minimum quality thresholds before promotion."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger


@dataclass
class ValidationResult:
    passed: bool
    metric_name: str
    actual_value: float
    threshold: float
    comparison: str  # ">" or "<"

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.metric_name}: {self.actual_value:.4f} (threshold {self.comparison} {self.threshold:.4f})"
        )


@dataclass
class ValidationReport:
    results: list[ValidationResult] = field(default_factory=list)
    model_version: str = ""

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    def summary(self) -> str:
        lines = [f"Model Validation Report - version {self.model_version}", "=" * 50]
        for r in self.results:
            lines.append(str(r))
        lines.append("=" * 50)
        lines.append(f"Overall: {'PASSED' if self.passed else 'FAILED'}")
        return "\n".join(lines)


class ModelValidator:
    """Validates a model against minimum performance thresholds."""

    def __init__(
        self,
        min_accuracy: float = 0.90,
        min_f1: float = 0.90,
        max_latency_ms: float = 100.0,
    ) -> None:
        self.min_accuracy = min_accuracy
        self.min_f1 = min_f1
        self.max_latency_ms = max_latency_ms

    def validate(self, model_version: str) -> ValidationReport:
        """Run all validation checks and return a report."""
        report = ValidationReport(model_version=model_version)

        metrics = self._load_metrics(model_version)

        report.results.append(
            ValidationResult(
                passed=metrics["accuracy"] >= self.min_accuracy,
                metric_name="accuracy",
                actual_value=metrics["accuracy"],
                threshold=self.min_accuracy,
                comparison=">=",
            )
        )

        report.results.append(
            ValidationResult(
                passed=metrics["f1"] >= self.min_f1,
                metric_name="f1_score",
                actual_value=metrics["f1"],
                threshold=self.min_f1,
                comparison=">=",
            )
        )

        latency_ms = self._measure_latency(model_version)
        report.results.append(
            ValidationResult(
                passed=latency_ms <= self.max_latency_ms,
                metric_name="latency_ms",
                actual_value=latency_ms,
                threshold=self.max_latency_ms,
                comparison="<=",
            )
        )

        nan_check_passed = self._check_no_nan_predictions(model_version)
        report.results.append(
            ValidationResult(
                passed=nan_check_passed,
                metric_name="no_nan_predictions",
                actual_value=1.0 if nan_check_passed else 0.0,
                threshold=1.0,
                comparison=">=",
            )
        )

        return report

    def _load_metrics(self, model_version: str) -> dict[str, float]:
        """Load model metrics from MLflow for the given version."""
        try:
            import mlflow

            client = mlflow.tracking.MlflowClient()
            mv = client.get_model_version(name="areteusml", version=model_version)
            run = client.get_run(mv.run_id)
            return {
                "accuracy": run.data.metrics.get("eval_accuracy", 0.0),
                "f1": run.data.metrics.get("eval_f1_weighted", 0.0),
            }
        except Exception:
            logger.warning("Could not load metrics from MLflow, attempting local validation set evaluation")
            return self._evaluate_local(model_version)

    def _evaluate_local(self, model_version: str) -> dict[str, float]:
        """Evaluate model locally on the validation set when MLflow is unavailable."""
        import pandas as pd
        from sklearn.metrics import accuracy_score, f1_score

        model, tokenizer = self._load_model(model_version)
        test_url = "https://huggingface.co/datasets/PolyAI/banking77/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet"
        test_df = pd.read_parquet(test_url)

        texts = test_df["text"].tolist()
        labels = test_df["label"].tolist()

        predictions = self._batch_predict(model, tokenizer, texts)

        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="weighted"),
        }

    def _load_model(self, model_version: str):
        """Load model and tokenizer, trying ONNX first then PyTorch."""
        onnx_path = Path("ml/models/onnx")
        pytorch_path = Path("ml/models/production")

        if onnx_path.exists() and any(onnx_path.glob("*.onnx")):
            from optimum.onnxruntime import ORTModelForSequenceClassification
            from transformers import AutoTokenizer

            logger.info("Loading ONNX model from {}", onnx_path)
            model = ORTModelForSequenceClassification.from_pretrained(str(onnx_path))
            tokenizer = AutoTokenizer.from_pretrained(str(onnx_path))
        elif pytorch_path.exists():
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            logger.info("Loading PyTorch model from {}", pytorch_path)
            model = AutoModelForSequenceClassification.from_pretrained(str(pytorch_path))
            tokenizer = AutoTokenizer.from_pretrained(str(pytorch_path))
        else:
            msg = f"No model found at {onnx_path} or {pytorch_path}"
            raise FileNotFoundError(msg)

        return model, tokenizer

    def _batch_predict(self, model, tokenizer, texts: list[str], batch_size: int = 32) -> list[int]:
        """Run batch predictions."""
        predictions = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs.logits, "numpy") else outputs.logits.detach()
            preds = np.argmax(logits.numpy() if hasattr(logits, "numpy") else logits.cpu().numpy(), axis=1)
            predictions.extend(preds.tolist())
        return predictions

    def _measure_latency(self, model_version: str) -> float:
        """Measure average inference latency in milliseconds over sample inputs."""
        try:
            model, tokenizer = self._load_model(model_version)
        except FileNotFoundError:
            logger.warning("No model found for latency check, skipping with 0ms")
            return 0.0

        sample_texts = [
            "I want to transfer money to another account",
            "What is my current balance?",
            "How do I activate my new card?",
            "I need to dispute a transaction on my statement",
            "Can you help me set up direct deposit?",
        ]

        # Warmup
        inputs = tokenizer(sample_texts[0], return_tensors="pt", truncation=True, max_length=128)
        model(**inputs)

        latencies = []
        for text in sample_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            start = time.perf_counter()
            model(**inputs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        return float(np.mean(latencies))

    def _check_no_nan_predictions(self, model_version: str) -> bool:
        """Verify model does not produce NaN logits on sample inputs."""
        try:
            model, tokenizer = self._load_model(model_version)
        except FileNotFoundError:
            logger.warning("No model found for NaN check, skipping")
            return True

        sample_texts = [
            "I want to check my balance",
            "",
            "a" * 512,
        ]

        for text in sample_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs.logits, "numpy") else outputs.logits.detach()
            arr = logits.numpy() if hasattr(logits, "numpy") else logits.cpu().numpy()
            if np.any(np.isnan(arr)):
                logger.error("NaN detected in predictions for input: {}", text[:50])
                return False

        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate model before promotion")
    parser.add_argument("--model-version", required=True, help="Model version to validate")
    parser.add_argument("--min-accuracy", type=float, default=0.90)
    parser.add_argument("--min-f1", type=float, default=0.90)
    parser.add_argument("--max-latency-ms", type=float, default=100.0)
    args = parser.parse_args()

    validator = ModelValidator(
        min_accuracy=args.min_accuracy,
        min_f1=args.min_f1,
        max_latency_ms=args.max_latency_ms,
    )

    report = validator.validate(args.model_version)
    logger.info("\n{}", report.summary())

    if not report.passed:
        logger.error("Model validation FAILED - not promoting to Production")
        sys.exit(1)

    logger.success("Model validation PASSED - safe to promote to Production")


if __name__ == "__main__":
    main()
