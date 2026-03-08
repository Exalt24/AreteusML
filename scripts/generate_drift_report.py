"""Generate Evidently drift report comparing reference and current predictions."""

from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    ref_path = PROJECT_ROOT / "artifacts" / "data" / "reference_predictions.parquet"
    cur_path = PROJECT_ROOT / "artifacts" / "data" / "current_predictions.parquet"
    output_path = PROJECT_ROOT / "dashboard" / "reports" / "evidently_report.html"

    print(f"Loading reference predictions from {ref_path}")
    ref_df = pd.read_parquet(ref_path)
    print(f"  {len(ref_df)} samples")

    print(f"Loading current predictions from {cur_path}")
    cur_df = pd.read_parquet(cur_path)
    print(f"  {len(cur_df)} samples")

    # Select numeric/categorical columns for drift analysis
    columns = ["predicted_label", "confidence"]
    ref_subset = ref_df[columns]
    cur_subset = cur_df[columns]

    print("Generating drift report...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_subset, current_data=cur_subset)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output_path))
    print(f"\nDrift report saved to {output_path}")


if __name__ == "__main__":
    main()
