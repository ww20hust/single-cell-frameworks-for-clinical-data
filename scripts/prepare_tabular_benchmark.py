from __future__ import annotations

import argparse
from pathlib import Path

from clinical_static_baseline_benchmark.data import (
    fit_standardizers,
    load_benchmark_split,
    save_prepared_split,
    transform_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a standardized lab/metabolomics benchmark split.")
    parser.add_argument("--input", required=True, help="Directory with train/test CSVs or an .npz bundle.")
    parser.add_argument("--output-dir", required=True, help="Output directory for raw and scaled CSV files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_split = load_benchmark_split(args.input)
    lab_scaler, metab_scaler = fit_standardizers(raw_split)
    scaled_split = transform_split(raw_split, lab_scaler, metab_scaler)
    save_prepared_split(raw_split, scaled_split, lab_scaler, metab_scaler, Path(args.output_dir))
    print(f"Prepared benchmark written to {args.output_dir}")


if __name__ == "__main__":
    main()
