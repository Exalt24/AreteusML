"""Export fine-tuned ModernBERT to ONNX with INT8 quantization and benchmark.

Run with:
    python -m ml.training.export_onnx
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from loguru import logger
from rich.console import Console
from rich.table import Table
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "ml" / "models" / "production"
ONNX_DIR = PROJECT_ROOT / "ml" / "models" / "onnx"
DATA_DIR = PROJECT_ROOT / "ml" / "data" / "processed"

N_BENCHMARK_SAMPLES = 100
console = Console()


def load_pytorch_model() -> tuple:
    """Load the fine-tuned PyTorch model and tokenizer."""
    logger.info(f"Loading PyTorch model from {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return model, tokenizer


def export_to_onnx(model, tokenizer) -> Path:
    """Export PyTorch model to ONNX format via torch.onnx.export."""
    onnx_fp32_dir = ONNX_DIR / "fp32"
    onnx_fp32_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting ONNX FP32 model to {onnx_fp32_dir}")

    # Create dummy input
    dummy = tokenizer("test input", return_tensors="pt", truncation=True, max_length=64)
    dummy_input = (dummy["input_ids"], dummy["attention_mask"])

    onnx_path = onnx_fp32_dir / "model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=18,
    )
    tokenizer.save_pretrained(onnx_fp32_dir)

    logger.success(f"ONNX FP32 model saved to {onnx_fp32_dir}")
    return onnx_fp32_dir


def quantize_int8(onnx_fp32_dir: Path) -> Path:
    """Apply INT8 dynamic quantization to the ONNX model."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    onnx_int8_dir = ONNX_DIR / "int8"
    onnx_int8_dir.mkdir(parents=True, exist_ok=True)

    src_model = onnx_fp32_dir / "model.onnx"
    dst_model = onnx_int8_dir / "model.onnx"

    # Pre-process: clear shape info to avoid inference conflicts
    model = onnx.load(str(src_model))
    while len(model.graph.value_info) > 0:
        model.graph.value_info.pop()
    preprocessed = onnx_fp32_dir / "model_preprocessed.onnx"
    onnx.save(model, str(preprocessed))
    del model

    logger.info("Applying INT8 dynamic quantization...")
    quantize_dynamic(
        model_input=str(preprocessed),
        model_output=str(dst_model),
        weight_type=QuantType.QInt8,
    )
    preprocessed.unlink()  # Clean up

    # Copy tokenizer files so the INT8 dir is self-contained
    import shutil

    for f in onnx_fp32_dir.iterdir():
        if f.name != "model.onnx":
            shutil.copy2(f, onnx_int8_dir / f.name)

    logger.success(f"ONNX INT8 model saved to {onnx_int8_dir}")
    return onnx_int8_dir


def _get_sample_texts(n: int = N_BENCHMARK_SAMPLES) -> list[str]:
    """Load sample texts from the test split for benchmarking."""
    import pandas as pd

    test_path = DATA_DIR / "test.parquet"
    if test_path.exists():
        df = pd.read_parquet(test_path)
        texts = df["text"].tolist()[:n]
    else:
        logger.warning("Test data not found, using synthetic samples")
        texts = ["I need to activate my card"] * n
    return texts


def benchmark_pytorch(model, tokenizer, texts: list[str]) -> float:
    """Benchmark PyTorch FP32 inference latency (ms per sample)."""
    device = next(model.parameters()).device
    total = 0.0
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
            start = time.perf_counter()
            model(**inputs)
            total += time.perf_counter() - start
    return (total / len(texts)) * 1000


def benchmark_onnx(model_dir: Path, tokenizer, texts: list[str]) -> float:
    """Benchmark ONNX model inference latency (ms per sample)."""
    import numpy as np
    import onnxruntime as ort

    onnx_path = model_dir / "model.onnx"
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    total = 0.0
    for text in texts:
        inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=64)
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        start = time.perf_counter()
        session.run(None, ort_inputs)
        total += time.perf_counter() - start
    return (total / len(texts)) * 1000


def run_benchmark(model, tokenizer, onnx_fp32_dir: Path, onnx_int8_dir: Path) -> dict:
    """Run latency benchmarks across all model variants."""
    texts = _get_sample_texts()
    logger.info(f"Benchmarking on {len(texts)} samples...")

    results = {
        "pytorch_fp32_ms": benchmark_pytorch(model, tokenizer, texts),
        "onnx_fp32_ms": benchmark_onnx(onnx_fp32_dir, tokenizer, texts),
        "onnx_int8_ms": benchmark_onnx(onnx_int8_dir, tokenizer, texts),
        "n_samples": len(texts),
    }

    results["speedup_onnx_fp32"] = round(results["pytorch_fp32_ms"] / results["onnx_fp32_ms"], 2)
    results["speedup_onnx_int8"] = round(results["pytorch_fp32_ms"] / results["onnx_int8_ms"], 2)

    return results


def display_results(results: dict) -> None:
    """Display benchmark results in a rich table."""
    table = Table(title="Inference Latency Benchmark", show_lines=True)
    table.add_column("Model Variant", style="cyan")
    table.add_column("Latency (ms/sample)", justify="right", style="green")
    table.add_column("Speedup vs PyTorch", justify="right", style="yellow")

    table.add_row(
        "PyTorch FP32",
        f"{results['pytorch_fp32_ms']:.2f}",
        "1.00x",
    )
    table.add_row(
        "ONNX FP32",
        f"{results['onnx_fp32_ms']:.2f}",
        f"{results['speedup_onnx_fp32']:.2f}x",
    )
    table.add_row(
        "ONNX INT8",
        f"{results['onnx_int8_ms']:.2f}",
        f"{results['speedup_onnx_int8']:.2f}x",
    )
    console.print(table)


def main() -> None:
    """Export model to ONNX, quantize, and benchmark."""
    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_pytorch_model()
    onnx_fp32_dir = export_to_onnx(model, tokenizer)
    onnx_int8_dir = quantize_int8(onnx_fp32_dir)

    results = run_benchmark(model, tokenizer, onnx_fp32_dir, onnx_int8_dir)
    display_results(results)

    benchmark_path = ONNX_DIR / "benchmark_results.json"
    with open(benchmark_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.success(f"Benchmark results saved to {benchmark_path}")


if __name__ == "__main__":
    main()
