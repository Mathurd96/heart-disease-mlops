#!/usr/bin/env python
"""Generate a SYNTHETIC stand-in dataset (same schema as the cleaned UCI data).

For offline smoke tests of the pipeline only. It is NOT the real dataset;
never report model results obtained from it.

    python scripts/make_synthetic_data.py --out /tmp/synthetic_heart.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heart.data import make_synthetic  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=303)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    make_synthetic(args.n).to_csv(args.out, index=False)
    print(f"Wrote synthetic data ({args.n} rows) to {args.out}")


if __name__ == "__main__":
    main()
