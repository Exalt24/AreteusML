"""Generate simulated current predictions with drift from reference predictions."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.labels import LABEL_NAMES


def main():
    ref_path = PROJECT_ROOT / "artifacts" / "data" / "reference_predictions.parquet"
    output_path = PROJECT_ROOT / "artifacts" / "data" / "current_predictions.parquet"

    print(f"Loading reference predictions from {ref_path}")
    df = pd.read_parquet(ref_path)
    print(f"  {len(df)} samples loaded")

    rng = np.random.default_rng(42)

    # Take random 80% subset
    n_samples = int(len(df) * 0.8)
    indices = rng.choice(len(df), size=n_samples, replace=False)
    indices.sort()
    current = df.iloc[indices].copy().reset_index(drop=True)
    print(f"  Sampled {len(current)} rows (80%)")

    # Add noise: randomly change ~5% of predicted_labels to adjacent labels
    n_noisy = int(len(current) * 0.05)
    noisy_indices = rng.choice(len(current), size=n_noisy, replace=False)
    labels = current["predicted_label"].values.copy()
    num_labels = len(LABEL_NAMES)

    for idx in noisy_indices:
        offset = rng.choice([-1, 1])
        labels[idx] = np.clip(labels[idx] + offset, 0, num_labels - 1)

    current["predicted_label"] = labels
    current["predicted_label_name"] = [LABEL_NAMES[label] for label in labels]

    # Reduce confidence by small random amount
    noise_factors = rng.uniform(0.85, 1.0, size=len(current))
    current["confidence"] = current["confidence"].values * noise_factors

    output_path.parent.mkdir(parents=True, exist_ok=True)
    current.to_parquet(output_path, index=False)
    print(f"\nSaved {len(current)} predictions to {output_path}")
    print("\nSummary:")
    print(f"  Mean confidence: {current['confidence'].mean():.4f}")
    print(f"  Labels changed:  {n_noisy} ({n_noisy/len(current)*100:.1f}%)")


if __name__ == "__main__":
    main()
