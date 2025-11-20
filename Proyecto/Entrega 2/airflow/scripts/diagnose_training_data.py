"""
Diagnostic script to inspect training matrices used by the DAG.
Run inside the Airflow container: 
    docker compose exec airflow-scheduler python /opt/airflow/scripts/diagnose_training_data.py
It will print shapes, null counts, constant columns, uniques per column (top), and label balance.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sodai_core import training

OUT = Path("/opt/airflow/logs/diagnose_training_data.json")

def summarize_df(df: pd.DataFrame, max_unique_show: int = 5):
    info = {}
    info["shape"] = df.shape
    info["dtypes"] = {c: str(dtype) for c, dtype in df.dtypes.items()}
    info["nulls"] = {c: int(df[c].isna().sum()) for c in df.columns}
    info["n_unique"] = {c: int(df[c].nunique(dropna=False)) for c in df.columns}
    # constant columns
    info["constant_columns"] = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    # show top uniques for small cardinality cols
    info["sample_uniques"] = {}
    for c in df.columns:
        if info["n_unique"][c] <= max_unique_show:
            info["sample_uniques"][c] = df[c].value_counts().to_dict()
    return info


def main():
    print("Preparing data (may be heavy)...")
    matrices = training.prepare_training_data_from_raw()

    report = {}
    for key in ["X_train", "X_valid", "X_test"]:
        df = matrices[key]
        report[key] = summarize_df(df)

    # label summaries
    y_train = matrices["y_train"]
    y_valid = matrices["y_valid"]
    y_test = matrices["y_test"]
    report["y_train"] = y_train.value_counts().to_dict()
    report["y_valid"] = y_valid.value_counts().to_dict()
    report["y_test"] = y_test.value_counts().to_dict()

    # Save report
    try:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT, "w") as f:
            json.dump(report, f, indent=2, default=int)
        print("Wrote report to", OUT)
    except Exception as e:
        print("Failed to write report:", e)

    # Print concise summary to stdout
    print("--- Summary ---")
    print("X_train shape:", matrices["X_train"].shape)
    print("y_train distribution:", matrices["y_train"].value_counts().to_dict())
    print("Constant columns in X_train:", report["X_train"]["constant_columns"]) 

if __name__ == '__main__':
    main()
