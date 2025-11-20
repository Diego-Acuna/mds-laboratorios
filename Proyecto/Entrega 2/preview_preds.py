#!/usr/bin/env python3
"""
Preview rápido del último archivo de predicciones en `airflow/data/predictions`.

Uso:
  python preview_preds.py                   # usa path por defecto
  python preview_preds.py path/to/preds.parquet
"""
import sys
import glob
from pathlib import Path
import pandas as pd


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "airflow/data/predictions/*.parquet"
    files = glob.glob(arg)
    if not files:
        print("No se encontraron archivos de predicciones con el patrón:", arg)
        return
    latest = sorted(files)[-1]
    print("Mostrando primeras filas de:", latest)
    df = pd.read_parquet(latest)
    print(df.head(20).to_string())


if __name__ == "__main__":
    main()
import pandas as pd, sys, glob
p = sys.argv[1] if len(sys.argv)>1 else 'airflow/data/predictions/*.parquet'
files = glob.glob(p)
if not files:
    print('No prediction files found:', p)
else:
    df = pd.read_parquet(files[-1])
    print(df.head(10))