"""Attention visualization for fine-tuned ModernBERT.

Extracts attention weights and creates token-level heatmaps showing
which tokens the model focuses on for classification.

Run with:
    python -m ml.explainability.attention_viz
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "ml" / "models" / "production"
DATA_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "ml" / "explainability" / "outputs"

N_SAMPLES = 8
MAX_LENGTH = 128


def load_model() -> tuple:
    """Load fine-tuned model with attention output enabled."""
    logger.info(f"Loading model from {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, output_attentions=True)
    model.eval()
    return model, tokenizer


def get_sample_texts(n: int = N_SAMPLES) -> list[str]:
    """Load diverse sample texts from the test split."""
    test_path = DATA_DIR / "test.parquet"
    if test_path.exists():
        df = pd.read_parquet(test_path)
        # Sample from different labels for diversity
        if "label" in df.columns:
            samples = df.groupby("label").first().reset_index()
            texts = samples["text"].tolist()[:n]
        else:
            texts = df["text"].tolist()[:n]
    else:
        logger.warning("Test data not found, using example texts")
        texts = [
            "I need to activate my new card",
            "Why was I charged twice for my purchase?",
            "How do I change my PIN number?",
            "I want to transfer money to another account",
        ]
    return texts[:n]


def extract_attention(model, tokenizer, text: str) -> tuple[np.ndarray, list[str], int]:
    """Extract attention weights for a single text.

    Returns
    -------
    attention : np.ndarray
        Averaged attention weights across all heads in the last layer,
        shape (seq_len,).
    tokens : list[str]
        Decoded tokens.
    pred_label : int
        Predicted class index.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)

    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.attentions is a tuple of (n_layers,) each (batch, heads, seq, seq)
    last_layer_attn = outputs.attentions[-1]  # (1, heads, seq, seq)
    # Average across heads, take CLS token row (index 0)
    cls_attn = last_layer_attn[0].mean(dim=0)[0]  # (seq_len,)
    cls_attn = cls_attn.cpu().numpy()

    # Normalize to [0, 1]
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    pred_label = outputs.logits.argmax(dim=-1).item()

    return cls_attn, tokens, pred_label


def plot_attention_heatmap(
    tokens: list[str],
    attention: np.ndarray,
    pred_label: int,
    label_map: dict | None,
    save_path: Path,
) -> None:
    """Create a horizontal token-level attention heatmap."""
    n_tokens = len(tokens)
    fig, ax = plt.subplots(figsize=(max(10, n_tokens * 0.6), 2.5))

    attn_2d = attention[:n_tokens].reshape(1, -1)
    im = ax.imshow(attn_2d, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(n_tokens))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
    ax.set_yticks([])

    label_str = label_map.get(pred_label, str(pred_label)) if label_map else str(pred_label)
    ax.set_title(f"Predicted: {label_str}", fontsize=10, pad=10)

    plt.colorbar(im, ax=ax, label="Attention Weight", shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def load_label_map() -> dict | None:
    """Attempt to load label mapping from the model config."""
    config_path = MODEL_DIR / "config.json"
    if config_path.exists():
        import json

        with open(config_path) as f:
            config = json.load(f)
        if "id2label" in config:
            return {int(k): v for k, v in config["id2label"].items()}
    return None


def main() -> None:
    """Generate attention visualizations for sample texts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model()
    label_map = load_label_map()
    texts = get_sample_texts()

    logger.info(f"Generating attention heatmaps for {len(texts)} samples...")

    for i, text in enumerate(texts):
        attention, tokens, pred_label = extract_attention(model, tokenizer, text)
        save_path = OUTPUT_DIR / f"attention_sample_{i}.png"
        plot_attention_heatmap(tokens, attention, pred_label, label_map, save_path)
        logger.info(
            f"Sample {i}: '{text[:50]}...' -> {label_map.get(pred_label, pred_label) if label_map else pred_label}"
        )

    logger.success(f"All attention visualizations saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
