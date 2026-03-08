"""Generate reference predictions from ONNX model on test set."""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.labels import LABEL_NAMES


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def main():
    test_path = PROJECT_ROOT / "ml" / "data" / "processed" / "test.parquet"
    model_path = PROJECT_ROOT / "ml" / "models" / "onnx" / "int8" / "model.onnx"
    tokenizer_path = PROJECT_ROOT / "ml" / "models" / "production"
    output_path = PROJECT_ROOT / "artifacts" / "data" / "reference_predictions.parquet"

    print(f"Loading test data from {test_path}")
    df = pd.read_parquet(test_path)
    print(f"  {len(df)} samples loaded")

    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    print(f"Loading ONNX model from {model_path}")
    session = ort.InferenceSession(str(model_path))
    input_names = [inp.name for inp in session.get_inputs()]
    print(f"  Input names: {input_names}")

    texts = df["text"].tolist()
    all_preds = []
    all_confs = []
    batch_size = 64

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="np")

        feed = {}
        for name in input_names:
            if name in encoded:
                feed[name] = encoded[name].astype(np.int64)

        logits = session.run(None, feed)[0]
        probs = softmax(logits)
        preds = np.argmax(probs, axis=-1)
        confs = np.max(probs, axis=-1)

        all_preds.extend(preds.tolist())
        all_confs.extend(confs.tolist())

        if (i // batch_size) % 10 == 0:
            print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)}")

    result = pd.DataFrame({
        "text": texts,
        "predicted_label": all_preds,
        "predicted_label_name": [LABEL_NAMES[p] for p in all_preds],
        "confidence": all_confs,
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f"\nSaved {len(result)} predictions to {output_path}")
    print("\nSummary:")
    print(f"  Mean confidence: {np.mean(all_confs):.4f}")
    print(f"  Min confidence:  {np.min(all_confs):.4f}")
    print(f"  Unique labels:   {len(set(all_preds))}")


if __name__ == "__main__":
    main()
