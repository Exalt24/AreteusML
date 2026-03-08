"""Inference service for text classification."""

import time
from dataclasses import dataclass

import numpy as np

from backend.app.core.model_loader import get_model


@dataclass
class PredictionResult:
    label: int
    confidence: float
    label_name: str
    latency_ms: float


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _get_label_name(label: int, label_map: dict) -> str:
    return label_map.get(str(label), f"class_{label}")


async def predict_single(text: str) -> PredictionResult:
    """Run inference on a single text."""
    model_data = get_model()
    tokenizer = model_data["tokenizer"]
    label_map = model_data.get("label_map", {})

    start = time.perf_counter()

    if model_data["backend"] == "onnx":
        inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=512, padding=True)
        session = model_data["session"]
        input_names = [i.name for i in session.get_inputs()]
        ort_inputs = {k: v.astype(np.int64) for k, v in inputs.items() if k in input_names}
        outputs = session.run(None, ort_inputs)
        logits = outputs[0]
    else:
        import torch

        pt_model = model_data["model"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = pt_model(**inputs)
        logits = outputs.logits.numpy()

    latency_ms = (time.perf_counter() - start) * 1000
    probs = _softmax(logits[0])
    label = int(np.argmax(probs))
    confidence = float(probs[label])

    return PredictionResult(
        label=label,
        confidence=round(confidence, 4),
        label_name=_get_label_name(label, label_map),
        latency_ms=round(latency_ms, 2),
    )


async def predict_batch(texts: list[str]) -> list[PredictionResult]:
    """Run inference on a batch of texts."""
    model_data = get_model()
    tokenizer = model_data["tokenizer"]
    label_map = model_data.get("label_map", {})

    results = []

    if model_data["backend"] == "onnx":
        session = model_data["session"]
        input_names = [i.name for i in session.get_inputs()]

        for text in texts:
            start = time.perf_counter()
            inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=512, padding=True)
            ort_inputs = {k: v.astype(np.int64) for k, v in inputs.items() if k in input_names}
            outputs = session.run(None, ort_inputs)
            logits = outputs[0]
            latency_ms = (time.perf_counter() - start) * 1000

            probs = _softmax(logits[0])
            label = int(np.argmax(probs))
            confidence = float(probs[label])
            results.append(
                PredictionResult(
                    label=label,
                    confidence=round(confidence, 4),
                    label_name=_get_label_name(label, label_map),
                    latency_ms=round(latency_ms, 2),
                )
            )
    else:
        import torch

        pt_model = model_data["model"]

        for text in texts:
            start = time.perf_counter()
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            with torch.no_grad():
                outputs = pt_model(**inputs)
            logits = outputs.logits.numpy()
            latency_ms = (time.perf_counter() - start) * 1000

            probs = _softmax(logits[0])
            label = int(np.argmax(probs))
            confidence = float(probs[label])
            results.append(
                PredictionResult(
                    label=label,
                    confidence=round(confidence, 4),
                    label_name=_get_label_name(label, label_map),
                    latency_ms=round(latency_ms, 2),
                )
            )

    return results
