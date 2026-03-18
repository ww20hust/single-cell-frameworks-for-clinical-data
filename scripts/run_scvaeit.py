from __future__ import annotations

import argparse
from pathlib import Path

from clinical_static_baseline_benchmark.data import (
    fit_standardizers,
    load_benchmark_split,
    save_prepared_split,
    transform_split,
)
from clinical_static_baseline_benchmark.io_utils import save_prediction_csv
from clinical_static_baseline_benchmark.metrics import evaluate_reconstruction, save_metrics
from clinical_static_baseline_benchmark.scvaeit_adapter import predict_scvaeit, train_scvaeit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scVAEIT on lab-to-metabolomics imputation.")
    parser.add_argument("--input", required=True, help="Directory with train/test CSVs or an .npz bundle.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--num-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    prepared_dir = output_dir / "prepared"
    model_dir = output_dir / "scvaeit_model"

    raw_split = load_benchmark_split(args.input)
    lab_scaler, metab_scaler = fit_standardizers(raw_split)
    scaled_split = transform_split(raw_split, lab_scaler, metab_scaler)
    save_prepared_split(raw_split, scaled_split, lab_scaler, metab_scaler, prepared_dir)

    checkpoint_dir = train_scvaeit(
        scaled_split.train_lab,
        scaled_split.train_metab,
        model_dir,
        random_seed=args.seed,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epoch=args.num_epoch,
        verbose=args.verbose,
    )
    pred_scaled = predict_scvaeit(
        checkpoint_dir,
        scaled_split.test_lab,
        metab_dim=scaled_split.train_metab.shape[1],
    )
    pred_raw = metab_scaler.inverse_transform(pred_scaled)

    save_prediction_csv(pred_scaled, output_dir / "test_metab_pred_scaled.csv", prefix="metab")
    save_prediction_csv(pred_raw, output_dir / "test_metab_pred.csv", prefix="metab")
    save_metrics(
        evaluate_reconstruction(raw_split.test_metab, pred_raw),
        output_dir / "metrics.json",
    )
    print(f"scVAEIT benchmark completed: {output_dir}")


if __name__ == "__main__":
    main()
