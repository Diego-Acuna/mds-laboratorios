#!/usr/bin/env python3
"""
Genera un fichero `transacciones_next_week.parquet` con una semana adicional
para simular la llegada de nuevos datos (útil para forzar drift y reentrenamiento).

Uso:
  python scripts/generate_new_week.py --data-dir airflow/data --n-sample 2000
  python scripts/generate_new_week.py --data-dir airflow/data --replace

Por seguridad el script por defecto NO sobrescribe el fichero original; crea
`transacciones_next_week.parquet`. Use --replace para moverlo a
`transacciones.parquet` (se hará un backup).
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import timedelta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="airflow/data", help="Directorio data de Airflow")
    p.add_argument("--input", default="transacciones.parquet", help="Archivo de transacciones de entrada")
    p.add_argument("--output", default="transacciones_next_week.parquet", help="Archivo de salida (en data-dir)")
    p.add_argument("--n-sample", type=int, default=2000, help="Número de filas nuevas (máximo)")
    p.add_argument("--replace", action="store_true", help="Si se pasa, reemplaza el archivo original (hace backup)")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    input_path = data_dir / args.input
    output_path = data_dir / args.output

    if not input_path.exists():
        raise SystemExit(f"No se encontró {input_path}; coloca aquí tu archivo transacciones.parquet")

    print(f"Cargando {input_path} ...")
    df = pd.read_parquet(input_path)

    if "purchase_date" not in df.columns:
        raise SystemExit("El archivo de transacciones debe tener la columna 'purchase_date'.")

    df["purchase_date"] = pd.to_datetime(df["purchase_date"]) 
    last_date = df["purchase_date"].max()
    new_date = last_date + timedelta(days=7)

    # Construir pares únicos customer-product
    if "customer_id" not in df.columns or "product_id" not in df.columns:
        raise SystemExit("El fichero debe contener 'customer_id' y 'product_id'.")

    unique_pairs = df[["customer_id", "product_id"]].drop_duplicates()
    n = min(args.n_sample, len(unique_pairs))
    sample = unique_pairs.sample(n=n, random_state=42).reset_index(drop=True)

    # Construir nuevas filas: mantenemos columnas si existen
    new_rows = sample.copy()
    new_rows["purchase_date"] = new_date

    # Añadir columnas típicas si están en el original
    for col in df.columns:
        if col not in new_rows.columns:
            if col == "payment":
                # asignar un pago positivo razonable
                new_rows[col] = 1.0
            elif col == "quantity":
                new_rows[col] = 1
            else:
                new_rows[col] = pd.NA

    # Append and save as a new parquet file
    combined = pd.concat([df, new_rows], ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)

    print(f"Archivo con semana nueva creado en: {output_path}")

    if args.replace:
        backup = input_path.with_suffix(input_path.suffix + ".bak")
        print(f"Haciendo backup {input_path} -> {backup} y reemplazando...")
        input_path.rename(backup)
        output_path.rename(input_path)
        print("Reemplazo completado. Ahora el DAG verá los nuevos datos.")


if __name__ == "__main__":
    main()
