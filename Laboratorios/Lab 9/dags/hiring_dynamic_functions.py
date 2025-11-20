from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Any, Dict, Tuple, Union, List

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.base import ClassifierMixin

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


# ---------------------------------------------------------------------
# Utilidades de paths y contexto (robustas fuera y dentro de Airflow)
# ---------------------------------------------------------------------

TARGET = "HiringDecision"

def _base_dir() -> Path:
    # carpeta donde vive este archivo (normalmente /opt/airflow/dags)
    return Path(__file__).resolve().parent

def _runs_dir() -> Path:
    return _base_dir() / "runs"

def _get_airflow_context(kwargs: Optional[dict]) -> Optional[dict]:
    try:
        # Airflow 2.x
        from airflow.operators.python import get_current_context  # type: ignore
        return get_current_context()
    except Exception:
        return kwargs if isinstance(kwargs, dict) else None

def _resolve_run_folder_name(kwargs: Optional[dict]) -> str:
    """
    Nombre de carpeta YYYYMMDD usando (en orden): ds_nodash, ds, hoy.
    """
    ctx = _get_airflow_context(kwargs)
    if ctx:
        ds_nodash = ctx.get("ds_nodash")
        if ds_nodash:
            return ds_nodash
        ds = ctx.get("ds")
        if ds:
            return ds.replace("-", "")
    return datetime.utcnow().strftime("%Y%m%d")

def _latest_run_dir() -> Optional[Path]:
    runs = sorted([p for p in _runs_dir().glob("*") if p.is_dir()], reverse=True)
    return runs[0] if runs else None

def _current_run_dir_from_xcom(kwargs: Optional[dict]) -> Optional[Path]:
    """
    Intenta recuperar el run_dir publicado por create_folders() vía XCom.
    Si no existe, usa la última corrida en runs/.
    """
    ctx = _get_airflow_context(kwargs)
    if ctx:
        ti = ctx.get("ti")
        if ti is not None:
            try:
                path_str = ti.xcom_pull(task_ids="create_folders")
                if path_str:
                    p = Path(path_str)
                    if p.exists():
                        return p
            except Exception:
                pass
    return _latest_run_dir()

def _ensure_dirs(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# create_folders
# ---------------------------------------------------------------------

def create_folders(**kwargs) -> str:
    """
    Crea carpeta de corrida y subcarpetas: raw, preprocessed, splits, models.
    Devuelve la ruta absoluta de la corrida (string).
    Publica por XCom si se usa con PythonOperator.
    """
    run_name = _resolve_run_folder_name(kwargs)
    run_dir = _runs_dir() / run_name
    for sub in ["", "raw", "preprocessed", "splits", "models"]:
        _ensure_dirs(run_dir / sub)

    print(f"[create_folders] Carpeta de corrida: {run_dir}")
    print(f"[create_folders] Subcarpetas: {[p.name for p in (run_dir).iterdir() if p.is_dir()]}")

    # devolver string para que Airflow lo pille en XCom por retorno
    return str(run_dir)


# ---------------------------------------------------------------------
# load_ands_merge
# ---------------------------------------------------------------------

def load_ands_merge(**kwargs) -> str:
    """
    Lee data_1.csv y, si existe, data_2.csv desde runs/<date>/raw.
    Concatena (stack vertical) y guarda en runs/<date>/preprocessed/merged.csv.
    Devuelve ruta del archivo resultante.
    """
    run_dir = _current_run_dir_from_xcom(kwargs)
    if not run_dir:
        raise FileNotFoundError("No se encontró carpeta de corrida. Ejecute create_folders primero.")

    raw = run_dir / "raw"
    out = run_dir / "preprocessed" / "merged.csv"

    paths = [raw / "data_1.csv", raw / "data_2.csv"]
    available = [p for p in paths if p.exists()]
    if not available:
        raise FileNotFoundError("No existe data_1.csv (ni data_2.csv) en la carpeta raw.")

    dfs = [pd.read_csv(p) for p in available]
    df = pd.concat(dfs, axis=0, ignore_index=True)

    # Opcional: limpieza mínima (no agresiva)
    # df = df.drop_duplicates()

    df.to_csv(out, index=False)
    print(f"[load_ands_merge] Archivos usados: {[p.name for p in available]}")
    print(f"[load_ands_merge] Guardado: {out} (shape={df.shape})")
    return str(out)

# Alias por si alguien llama con el nombre “natural”
load_and_merge = load_ands_merge


# ---------------------------------------------------------------------
# split_data
# ---------------------------------------------------------------------

def split_data(test_size: float = 0.20, random_state: int = 42, **kwargs) -> Tuple[str, str]:
    """
    Lee preprocessed/merged.csv, realiza hold-out estratificado y guarda:
      - splits/train.csv
      - splits/test.csv
    Devuelve (train_path, test_path).
    """
    run_dir = _current_run_dir_from_xcom(kwargs)
    if not run_dir:
        raise FileNotFoundError("No hay carpeta de corrida disponible.")

    pre = run_dir / "preprocessed" / "merged.csv"
    if not pre.exists():
        raise FileNotFoundError("No existe preprocessed/merged.csv. Ejecute load_ands_merge primero.")

    df = pd.read_csv(pre)
    if TARGET not in df.columns:
        raise ValueError(f"No se encontró la variable objetivo '{TARGET}' en el dataset.")

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    splits_dir = run_dir / "splits"
    _ensure_dirs(splits_dir)
    tr_path = splits_dir / "train.csv"
    te_path = splits_dir / "test.csv"

    pd.concat([Xtr, ytr.rename(TARGET)], axis=1).to_csv(tr_path, index=False)
    pd.concat([Xte, yte.rename(TARGET)], axis=1).to_csv(te_path, index=False)

    print(f"[split_data] Train: {tr_path} (n={len(Xtr)}) | Test: {te_path} (n={len(Xte)})")
    return str(tr_path), str(te_path)


# ---------------------------------------------------------------------
# train_model
# ---------------------------------------------------------------------

# Mapeo de nombres → estimadores (si prefieres pasar un string)
_ESTIMATORS = {
    "random_forest": RandomForestClassifier,
    "logreg": LogisticRegression,
    "gbc": GradientBoostingClassifier,
}

def _build_preprocessor(columns: List[str]) -> ColumnTransformer:
    """
    Define transformaciones por tipo. Categóricas (codificadas como enteros):
      Gender, EducationLevel, RecruitmentStrategy
    El resto se tratan como numéricas.
    """
    cat_candidates = {"Gender", "EducationLevel", "RecruitmentStrategy"}
    cats = sorted([c for c in columns if c in cat_candidates])
    nums = sorted([c for c in columns if c not in cat_candidates])

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, nums),
            ("cat", cat_pipe, cats),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        n_jobs=None,
    )
    print(f"[train_model] Numéricas: {nums}")
    print(f"[train_model] Categóricas: {cats}")
    return pre


def train_model(
    estimator: Union[str, ClassifierMixin] = "random_forest",
    estimator_params: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> str:
    """
    Entrena un Pipeline(preprocesado + modelo) con el conjunto de entrenamiento.
    - estimator: puede ser un string ('random_forest','logreg','gbc') o un estimador sklearn.
    - estimator_params: dict opcional de hiperparámetros (cuando estimator es string).

    Guarda el modelo en runs/<date>/models/model_<nombre>_<timestamp>.joblib
    Devuelve la ruta del .joblib.
    """
    run_dir = _current_run_dir_from_xcom(kwargs)
    if not run_dir:
        raise FileNotFoundError("No hay carpeta de corrida disponible.")

    train_path = run_dir / "splits" / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError("No existe splits/train.csv. Ejecute split_data primero.")

    train_df = pd.read_csv(train_path)
    if TARGET not in train_df.columns:
        raise ValueError(f"'{TARGET}' no está en train.csv")

    y = train_df[TARGET].astype(int)
    X = train_df.drop(columns=[TARGET])

    # Construye el estimador
    if isinstance(estimator, str):
        est_cls = _ESTIMATORS.get(estimator.lower())
        if est_cls is None:
            raise ValueError(f"Modelo '{estimator}' no soportado. Use: {list(_ESTIMATORS.keys())}")
        est = est_cls(**(estimator_params or {}))
        est_name = estimator.lower()
    else:
        est = estimator
        est_name = est.__class__.__name__.lower()

    pre = _build_preprocessor(list(X.columns))
    pipe = Pipeline([
        ("pre", pre),
        ("clf", est),
    ])

    pipe.fit(X, y)

    models_dir = run_dir / "models"
    _ensure_dirs(models_dir)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out = models_dir / f"model_{est_name}_{ts}.joblib"
    joblib.dump(pipe, out)

    print(f"[train_model] Modelo entrenado: {est.__class__.__name__}")
    print(f"[train_model] Guardado en: {out}")
    return str(out)


# ---------------------------------------------------------------------
# evaluate_models
# ---------------------------------------------------------------------

def evaluate_models(**kwargs) -> str:
    """
    Carga todos los .joblib de runs/<date>/models, evalúa accuracy en el test set
    y selecciona el mejor. Guarda una copia como best_model.joblib.
    Imprime el nombre del mejor y su accuracy. Devuelve la ruta del best_model.
    """
    run_dir = _current_run_dir_from_xcom(kwargs)
    if not run_dir:
        raise FileNotFoundError("No hay carpeta de corrida disponible.")

    test_path = run_dir / "splits" / "test.csv"
    models_dir = run_dir / "models"
    if not test_path.exists():
        raise FileNotFoundError("No existe splits/test.csv. Ejecute split_data primero.")
    if not models_dir.exists():
        raise FileNotFoundError("No existe carpeta models/. Entrene al menos un modelo.")

    test_df = pd.read_csv(test_path)
    if TARGET not in test_df.columns:
        raise ValueError(f"'{TARGET}' no está en test.csv")

    y_true = test_df[TARGET].astype(int).to_numpy()
    X_test = test_df.drop(columns=[TARGET])

    model_paths = sorted([p for p in models_dir.glob("model_*.joblib") if p.is_file()])
    if not model_paths:
        raise FileNotFoundError("No se encontraron modelos para evaluar en models/.")

    # Evalúa todos los modelos
    scores = []
    for mp in model_paths:
        try:
            pipe = joblib.load(mp)
            y_pred = pipe.predict(X_test)
            acc = accuracy_score(y_true, y_pred)
            scores.append((acc, mp))
            print(f"[evaluate_models] {mp.name}: accuracy={acc:.4f}")
        except Exception as e:
            print(f"[evaluate_models] Falló {mp.name}: {e}")

    if not scores:
        raise RuntimeError("No fue posible evaluar ningún modelo.")

    best_acc, best_path = max(scores, key=lambda t: t[0])
    best_model_copy = models_dir / "best_model.joblib"
    joblib.dump(joblib.load(best_path), best_model_copy)

    print(f"[evaluate_models] Mejor modelo: {best_path.name} | accuracy={best_acc:.4f}")
    print(f"[evaluate_models] Guardado como: {best_model_copy}")
    return str(best_model_copy)