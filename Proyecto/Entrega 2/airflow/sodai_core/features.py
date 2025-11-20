# airflow/sodai_core/features.py

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, List
from . import data_io

import numpy as np
import pandas as pd


# ======================
# Configuración de columnas
# ======================

TARGET_COL = "bought"

# Estas listas deben coincidir con las se usaron en la entrega 1
NUM_FEATURES: List[str] = [
    "cp_recency",
    "cp_prev_bought",
    "cp_count_4w",
    "cp_count_12w",
    "prod_prior_rate",
    "cust_cat_prior_rate",
    "size_liters",
    "num_deliver_per_week",
    "num_visit_per_week",
    "x",
    "y",
]

CAT_FEATURES: List[str] = [
    "region_id",
    "zone_id",
    "customer_type",
    "brand",
    "category",
    "sub_category",
    "segment",
    "package",
]

ID_COLS: List[str] = [
    "customer_id",
    "product_id",
    "week",
]


# ======================
# Limpieza de transacciones
# ======================

def clean_transactions(transacciones: pd.DataFrame) -> pd.DataFrame:
    """
    Replica la lógica de generación de `transacciones_limpio`
    implementada en el notebook de la entrega 1.
    """
    df = transacciones.copy()

    # 1) parseo de fecha
    df["purchase_date"] = pd.to_datetime(df["purchase_date"])

    # 2) filtrado de nulos básicos
    df = df.dropna(subset=["customer_id", "product_id", "purchase_date"])

    # 3) filtrar pagos no positivos
    if "payment" in df.columns:
        df = df[df["payment"] > 0]

    # 4) eliminar duplicados
    subset_dup = ["customer_id", "product_id", "purchase_date"]
    df = df.drop_duplicates(subset=subset_dup)

    return df.reset_index(drop=True)


# ======================
# Construcción de interacciones semanales
# ======================

def build_weekly_interactions(
    transacciones_limpio: pd.DataFrame,
    clientes: pd.DataFrame,
    productos: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye la tabla `interacciones_semana` a partir de transacciones limpias,
    uniendo con clientes y productos e incorporando la granularidad semanal.
    """
    df = transacciones_limpio.copy()

    # Semana (puedes cambiar la regla si usaste otra en el notebook)
    week = df["purchase_date"].dt.to_period("W-MON").dt.start_time
    df["week"] = week

    # Agregamos a nivel cliente-producto-semana (solo donde hay transacciones)
    group_cols = ["customer_id", "product_id", "week"]

    # size() no necesita ninguna columna específica, solo cuenta filas
    agg = df.groupby(group_cols, as_index=False).size()

    # La columna que crea size() se llama "size" -> la renombramos
    agg = agg.rename(columns={"size": "num_tx"})

    # bought = 1 si hay al menos una transacción esa semana
    agg["bought"] = 1

    # ------------------------------------------------------------------
    # Estrategia 2A: generar candidatos por cliente de forma eficiente
    # - Para cada cliente, tomar los productos que compró históricamente
    # - Añadir además top-K productos globales para cubrir nuevos items
    # - Luego expandir por semanas (cross join con semanas) — esto es
    #   mucho más pequeño que el producto cartesiano completo.
    # ------------------------------------------------------------------
    TOP_K = 20

    # clientes únicos
    clientes_ids = clientes[["customer_id"]].drop_duplicates().reset_index(drop=True)

    # productos comprados históricamente por cliente
    cust_prod = (
        transacciones_limpio[["customer_id", "product_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # top-K productos globales por frecuencia
    top_products = (
        transacciones_limpio["product_id"].value_counts().nlargest(TOP_K).index.tolist()
    )
    top_df = pd.DataFrame({"product_id": top_products})

    # cross join clientes x top_products to ensure every customer has top-K candidates
    clientes_ids["_key"] = 1
    top_df["_key"] = 1
    cust_top = clientes_ids.merge(top_df, on="_key").drop(columns=["_key"])

    # unir productos históricos + top-K, por cliente
    candidate_prod = pd.concat([cust_prod, cust_top], ignore_index=True).drop_duplicates()

    # semanas observadas en los datos
    weeks = pd.Series(sorted(df["week"].drop_duplicates()))
    weeks_df = pd.DataFrame({"week": weeks}).reset_index(drop=True)

    # cross join candidato cliente-producto con semanas
    candidate_prod["_key"] = 1
    weeks_df["_key"] = 1
    cart = candidate_prod.merge(weeks_df, on="_key").drop(columns=["_key"])  # customer_id, product_id, week

    # Left join de las agregaciones (transacciones) sobre el universo reducido
    out = cart.merge(agg, on=["customer_id", "product_id", "week"], how="left")

    # Rellenar valores donde no hubo transacción
    out["num_tx"] = out["num_tx"].fillna(0).astype(int)
    out["bought"] = out["bought"].fillna(0).astype(int)

    # Join con clientes y productos para añadir metadata
    clientes_ren = clientes.rename(columns={"X": "x", "Y": "y"})
    productos_ren = productos.copy()

    out = (
        out
        .merge(clientes_ren, on="customer_id", how="left")
        .merge(productos_ren, on="product_id", how="left")
    )

    # size_liters desde la columna size de productos (si existe)
    if "size" in out.columns and "size_liters" not in out.columns:
        size_str = out["size"].astype(str).str.lower().str.replace("l", "", regex=False)
        size_str = size_str.str.replace(",", ".", regex=False)
        with np.errstate(invalid="ignore"):
            out["size_liters"] = pd.to_numeric(size_str, errors="coerce")

    return out



# ======================
# Ingeniería de lags y recency
# ======================

def add_behavioral_features(interacciones_semana: pd.DataFrame) -> pd.DataFrame:
    """
    Añade recency, conteos móviles y tasas históricas (cp_recency, cp_prev_bought,
    cp_count_4w, cp_count_12w, prod_prior_rate, cust_cat_prior_rate, etc.).
    """
    df = interacciones_semana.copy()

    df = df.sort_values(["customer_id", "product_id", "week"])

    grp_cp = df.groupby(["customer_id", "product_id"], sort=False)

    # cp_prev_bought: compras acumuladas antes de la semana actual
    df["cp_prev_bought"] = grp_cp["bought"].cumsum() - df["bought"]

    # Índice entero de la semana dentro de cada grupo (0, 1, 2, ...)
    df["cp_week_idx"] = grp_cp.cumcount()

    # Recency (en semanas) desde la última compra dentro del grupo cliente-producto.
    bought = df["bought"].to_numpy().astype(bool)
    idx = df["cp_week_idx"].to_numpy()

    group_ids = grp_cp.grouper.group_info[0]
    # Usamos un truco vectorizado por grupo:
    last_buy_idx = np.full_like(idx, -1)

    for g in np.unique(group_ids):
        mask_g = group_ids == g
        idx_g = idx[mask_g]
        bought_g = bought[mask_g]

        last = np.where(bought_g, idx_g, -1)
        last = np.maximum.accumulate(last)
        last_buy_idx[mask_g] = last

    recency = (idx - last_buy_idx).astype("float32")
    recency[last_buy_idx == -1] = np.nan
    df["cp_recency"] = recency

    # Ventanas móviles de 4 y 12 semanas (aproximación: rolling por índice de grupo)
    df["cp_count_4w"] = (
        grp_cp["bought"].rolling(window=4, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
    )
    df["cp_count_12w"] = (
        grp_cp["bought"].rolling(window=12, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
    )

    # Tasa histórica por producto
    grp_prod = df.groupby("product_id", sort=False)
    prior_buys = grp_prod["bought"].cumsum().shift(1).fillna(0.0)
    prior_obs = grp_prod["bought"].expanding().count().shift(1).fillna(0.0)
    prior_buys_arr = np.asarray(prior_buys, dtype="float32")
    prior_obs_arr = np.asarray(prior_obs, dtype="float32")

    df["prod_prior_rate"] = ((prior_buys_arr + 1.0) / (prior_obs_arr + 2.0)).astype("float32")

    # Tasa histórica cliente × categoría
    if "category" in df.columns:
        grp_cust_cat = df.groupby(["customer_id", "category"], sort=False)
        prior_buys_cc = grp_cust_cat["bought"].cumsum().shift(1).fillna(0.0)
        prior_obs_cc = grp_cust_cat["bought"].expanding().count().shift(1).fillna(0.0)
        prior_buys_cc_arr = np.asarray(prior_buys_cc, dtype="float32")
        prior_obs_cc_arr = np.asarray(prior_obs_cc, dtype="float32")
        df["cust_cat_prior_rate"] = ((prior_buys_cc_arr + 1.0) / (prior_obs_cc_arr + 2.0)).astype("float32")
    else:
        df["cust_cat_prior_rate"] = np.nan

    return df


# ======================
# Holdout temporal
# ======================

def temporal_holdout(
    interacciones_semana: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Realiza la partición temporal train/valid/test.
    """
    df = interacciones_semana.copy()

    weeks = np.sort(df["week"].unique())
    weeks = pd.to_datetime(weeks)

    # Implementación simple basada en 35/7/9 semanas y dejando gaps de 1 semana.
    if len(weeks) < 51:
        raise ValueError("Se esperaban al menos ~51 semanas para replicar el holdout descrito.")

    train_weeks = weeks[:35]
    valid_weeks = weeks[36:36 + 7]
    test_weeks = weeks[36 + 7 + 1:]

    df["split"] = "test"
    df.loc[df["week"].isin(train_weeks), "split"] = "train"
    df.loc[df["week"].isin(valid_weeks), "split"] = "valid"

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    valid_df = df[df["split"] == "valid"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    return train_df, valid_df, test_df


# ======================
# Construcción de matrices de entrenamiento
# ======================

def build_ml_matrices(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Construye las matrices X/y para train, valid y test.
    """
    cols_features = NUM_FEATURES + CAT_FEATURES

    def _split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        x = df[cols_features].copy()
        y = df[TARGET_COL].astype(int).copy()
        ids = df[ID_COLS].copy()
        return x, y, ids

    X_train, y_train, ids_train = _split_xy(df_train)
    X_valid, y_valid, ids_valid = _split_xy(df_valid)
    X_test, y_test, ids_test = _split_xy(df_test)

    return dict(
        X_train=X_train,
        y_train=y_train,
        ids_train=ids_train,
        X_valid=X_valid,
        y_valid=y_valid,
        ids_valid=ids_valid,
        X_test=X_test,
        y_test=y_test,
        ids_test=ids_test,
    )
    
def build_feature_matrix_from_parquets(
    data_dir: Path | str,
    output_path: Path | str,
) -> Path:
    """
    Función de alto nivel usada por el DAG.

    - Carga los .parquet crudos (transacciones, clientes, productos).
    - Limpia transacciones.
    - Construye interacciones semanales.
    - Añade features de comportamiento.
    - Guarda el resultado en un parquet (feature_matrix_latest).

    Devuelve la ruta donde se guardó la matriz.
    """
    data_dir = Path(data_dir)
    output_path = Path(output_path)

    # 1) Cargar datos crudos
    transacciones, clientes, productos = data_io.load_raw_tables(data_dir)

    # 2) Limpiar transacciones
    transacciones_limpio = clean_transactions(transacciones)

    # 3) Construir interacciones semanales
    interacciones = build_weekly_interactions(
        transacciones_limpio,
        clientes,
        productos,
    )

    # 4) Añadir features de comportamiento
    interacciones = add_behavioral_features(interacciones)

    # 5) Guardar a disco
    output_path.parent.mkdir(parents=True, exist_ok=True)
    interacciones.to_parquet(output_path, index=False)

    return output_path