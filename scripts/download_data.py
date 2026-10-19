#!/usr/bin/env python
"""Download the UCI Heart Disease (Cleveland) data and produce the cleaned CSV.

    python scripts/download_data.py

Outputs
    data/raw/processed.cleveland.data   (untouched UCI file)
    data/processed/heart_clean.csv      (cleaned, binary `target` column)

Manual alternative if you have no internet access from this machine:
    1. Open https://archive.ics.uci.edu/dataset/45/heart+disease
    2. Download the zip, extract `processed.cleveland.data` into data/raw/
    3. Run:  python scripts/download_data.py --skip-download
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heart import config, data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="reuse an existing data/raw/processed.cleveland.data",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.skip_download:
        data.download_raw()
    elif not config.RAW_FILE.exists():
        sys.exit(f"{config.RAW_FILE} not found; place the UCI file there first.")

    df = data.prepare_dataset()
    print(f"\nCleaned dataset: {len(df)} rows x {df.shape[1]} columns")
    print(f"Missing values per column:\n{df.isna().sum()[df.isna().sum() > 0].to_string()}")
    print(f"Class balance (target):\n{df[config.TARGET].value_counts().to_string()}")


if __name__ == "__main__":
    main()
