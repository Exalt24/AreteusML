# ADR-001: ModernBERT as Base Model

## Context

Need a base model for Banking77 77-class intent classification. Candidates evaluated: DeBERTa-v3, DistilBERT, ModernBERT-base.

## Decision

Use ModernBERT-base.

## Consequences

- 2-4x faster than DeBERTa-v3 with comparable accuracy, thanks to built-in FlashAttention and unpadding
- 8192 token context window (vs 512 for most BERT variants), future-proofs for longer inputs
- Released December 2024, showing we pick current tooling not legacy defaults
- Fits comfortably in 6GB VRAM with FP16, keeping training accessible on consumer hardware
