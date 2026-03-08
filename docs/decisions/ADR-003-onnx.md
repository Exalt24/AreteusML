# ADR-003: ONNX Runtime for Inference

## Context

Need optimized inference for production serving. Candidates evaluated: TorchServe, Triton Inference Server, ONNX Runtime.

## Decision

Use ONNX Runtime with INT8 quantization.

## Consequences

- CPU-optimized inference eliminates GPU cost in production, making deployment viable on any server
- INT8 dynamic quantization targets <10ms latency per request, well under interactive thresholds
- Single-file model export (`.onnx`) simplifies deployment vs TorchServe's model archiver or Triton's model repository
- Triton is better for multi-model GPU serving at scale; ONNX Runtime is right-sized for single-model CPU deployment
