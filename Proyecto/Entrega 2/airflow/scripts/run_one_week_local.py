"""Run the scoring pipeline on a small, single-week subset of the raw data.

This script creates a temporary data directory containing a filtered
`transacciones.parquet` (and the original `clientes.parquet` &
`productos.parquet`) covering only the target week plus `--weeks-back`
weeks of history. It then calls `run_scoring_pipeline(...)` from
`airflow.sodai_core.inference` and prints/saves the resulting ranking.

Usage (from repo root):
  python airflow/scripts/run_one_week_local.py [--week YYYY-MM-DD] [--weeks-back 12] [--top-k 10]

Examples:
  # Use the latest available week and include 12 weeks history (default)
  python airflow/scripts/run_one_week_local.py

  # Specify a reference week
  python airflow/scripts/run_one_week_local.py --week 2024-11-04 --weeks-back 8 --top-k 5
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
import sys
import shutil
import pandas as pd

# Ensure we can import `airflow.sodai_core` when running from repo root
THIS_DIR = Path(__file__).resolve().parents[2]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from airflow.sodai_core import data_io, inference


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--week", type=str, default=None, help="Reference week (YYYY-MM-DD). If omitted, use latest week in the data")
    p.add_argument("--weeks-back", type=int, default=12, help="How many weeks of history to include (default: 12)")
    p.add_argument("--top-k", type=int, default=10, help="How many recommendations per customer to return (default: 10)")
    p.add_argument("--out", type=str, default=None, help="Optional output parquet path to save ranking")
    return p.parse_args()


def main():
    args = parse_args()

    data_dir = data_io.DATA_DIR
    print(f"Loading raw tables from {data_dir}")
    trans, clientes, productos = data_io.load_raw_tables(data_dir)

    # Ensure purchase_date is datetime and compute weekly period start (W-MON)
    trans = trans.copy()
    trans["purchase_date"] = pd.to_datetime(trans["purchase_date"])  # idempotent
    trans["week"] = trans["purchase_date"].dt.to_period("W-MON").dt.start_time

    if args.week is None:
        ref_week = trans["week"].max()
        print(f"Inferred latest week in data: {ref_week.date()}")
    else:
        ref_week = pd.to_datetime(args.week)
        # normalize to week start (W-MON)
        ref_week = ref_week.to_period("W-MON").start_time
        print(f"Using provided reference week: {ref_week.date()}")

    # Build a filtered transactions set including `weeks_back` weeks before ref_week
    weeks_back = int(args.weeks_back)
    start_window = ref_week - pd.Timedelta(weeks=weeks_back)
    end_window = ref_week + pd.Timedelta(days=6)
    print(f"Filtering transactions between {start_window.date()} and {end_window.date()} ({weeks_back} weeks history)")

    mask = (trans["purchase_date"] >= pd.Timestamp(start_window)) & (trans["purchase_date"] <= pd.Timestamp(end_window))
    trans_f = trans.loc[mask].reset_index(drop=True)
    print(f"Filtered transactions: {len(trans_f)} rows")

    # Create a temporary data directory and write filtered tables
    tmpdir = Path(tempfile.mkdtemp(prefix="sodai_one_week_"))
    print(f"Writing temporary data to {tmpdir}")
    try:
        (tmpdir / "transacciones.parquet").parent.mkdir(parents=True, exist_ok=True)
        trans_f.to_parquet(tmpdir / "transacciones.parquet", index=False)
        # Use full clients/products tables to preserve metadata
        clientes.to_parquet(tmpdir / "clientes.parquet", index=False)
        productos.to_parquet(tmpdir / "productos.parquet", index=False)

        # Run scoring pipeline using the temporary data dir
        print("Running scoring pipeline on filtered data (this should be fast)...")
        ranking = inference.run_scoring_pipeline(data_dir=tmpdir, reference_week=ref_week, top_k=args.top_k)

        print("Top rows of resulting ranking:")
        print(ranking.head(20).to_string(index=False))

        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            ranking.to_parquet(out_path, index=False)
            print(f"Saved ranking to {out_path}")
        else:
            # Save into the tempdir for easier retrieval if needed
            out_path = tmpdir / "predictions_one_week.parquet"
            ranking.to_parquet(out_path, index=False)
            print(f"Saved ranking to {out_path}")

    finally:
        print(f"Temporary data directory left at: {tmpdir} (remove when done)")


if __name__ == "__main__":
    main()
