from __future__ import annotations

import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier


# --- Utilidades para contexto Airflow (no rompe fuera de Airflow) ---
def _get_airflow_context() -> Optional[dict]:
    """Intenta obtener el contexto actual de Airflow cuando se ejecuta dentro de un DAG."""
    try:
        from airflow.operators.python import get_current_context  # Airflow 2.x
        return get_current_context()
    except Exception:
        return None


def _resolve_run_folder_name(kwargs: Optional[dict]) -> str:
    """
    Nombre YYYYMMDD para la carpeta de la corrida usando, en orden:
    kwargs['ds_nodash'] -> kwargs['ds'] -> kwargs['logical_date'] -> hoy UTC.
    """
    ds = None
    if kwargs:
        ds = kwargs.get("ds_nodash") or (kwargs.get("ds") or "").replace("-", "")
        if not ds and kwargs.get("logical_date"):
            try:
                ds = kwargs["logical_date"].strftime("%Y%m%d")
            except Exception:
                pass

    if not ds:
        ctx = _get_airflow_context()
        if ctx:
            ds = (ctx.get("ds_nodash") or (ctx.get("ds") or "").replace("-", "")) or (
                ctx.get("logical_date").strftime("%Y%m%d") if ctx.get("logical_date") else None
            )

    return ds or datetime.utcnow().strftime("%Y%m%d")


# --- Rutas y constantes de trabajo (bajo la carpeta dags) ---
THIS_DIR = Path(__file__).resolve().parent
RUNS_ROOT = THIS_DIR / "runs"         # runs/YYYYMMDD/{raw,splits,models}
DATASETS_DIR = THIS_DIR / "datasets"  # datasets/data_1.csv  (inbox opcional)

TARGET = "HiringDecision"
RAW_FILENAME = "data_1.csv"
MODEL_FILENAME = "rf_pipeline.joblib"


def _run_dir(kwargs: Optional[dict]) -> Path:
    """Devuelve la carpeta de la corrida: dags/runs/YYYYMMDD"""
    return RUNS_ROOT / _resolve_run_folder_name(kwargs)


# ---------------------------------------------------------------------
# 1) create_folders()
# ---------------------------------------------------------------------
def create_folders(**kwargs) -> str:
    """
    Crea la carpeta de corrida según la fecha de ejecución y sus subcarpetas:
    raw/, splits/, models/.
    """
    run_dir = _run_dir(kwargs)
    for sub in ("raw", "splits", "models"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    print(f"[create_folders] Carpeta de corrida creada: {run_dir}")
    print(f"[create_folders] Subcarpetas: {[p.name for p in run_dir.iterdir() if p.is_dir()]}")
    return str(run_dir)


# ---------------------------------------------------------------------
# 2) split_data()
# ---------------------------------------------------------------------
def split_data(**kwargs) -> str:
    """
    Lee dags/runs/YYYYMMDD/raw/data_1.csv, aplica holdout 80/20 estratificado,
    y guarda train.csv / test.csv en splits/.
    Si el CSV no está en raw/, intenta copiarlo desde dags/datasets/data_1.csv.
    """
    run_dir = _run_dir(kwargs)
    raw_path = run_dir / "raw" / RAW_FILENAME
    fallback = DATASETS_DIR / RAW_FILENAME
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists() and fallback.exists():
        shutil.copy(fallback, raw_path)
        print(f"[split_data] Copiado {fallback} -> {raw_path}")

    if not raw_path.exists():
        raise FileNotFoundError(
            f"No se encontró {raw_path}. "
            f"Copia 'data_1.csv' a esa ruta o a {fallback} y reintenta."
        )

    df = pd.read_csv(raw_path)
    if TARGET not in df.columns:
        raise ValueError(f"La columna objetivo '{TARGET}' no está en el CSV. Columnas: {df.columns.tolist()}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    splits_dir = run_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Guardamos sin bucles, de forma vectorizada
    X_train.assign(**{TARGET: y_train}).to_csv(splits_dir / "train.csv", index=False)
    X_test.assign(**{TARGET: y_test}).to_csv(splits_dir / "test.csv", index=False)

    print(f"[split_data] train/test guardados en {splits_dir}")
    return str(splits_dir)


# ---------------------------------------------------------------------
# 3) preprocess_and_train()
# ---------------------------------------------------------------------
def preprocess_and_train(**kwargs) -> str:
    """
    Lee splits/train.csv y splits/test.csv, arma un Pipeline con:
      - ColumnTransformer: StandardScaler (num), OneHotEncoder (cat)
      - RandomForestClassifier
    Entrena, evalúa (accuracy y f1 de la clase positiva=1), y persiste joblib.
    """
    run_dir = _run_dir(kwargs)
    splits_dir = run_dir / "splits"
    train_path = splits_dir / "train.csv"
    test_path = splits_dir / "test.csv"

    if not (train_path.exists() and test_path.exists()):
        raise FileNotFoundError("No existen los archivos de splits. Ejecuta primero 'split_data'.")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    y_train = train[TARGET].astype(int)
    y_test = test[TARGET].astype(int)
    X_train = train.drop(columns=[TARGET])
    X_test = test.drop(columns=[TARGET])

    # Categóricas ordinales/codificadas: las tratamos como nominales con OHE
    cat_cols = [c for c in ["Gender", "EducationLevel", "RecruitmentStrategy"] if c in X_train.columns]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    preproc = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop"
    )

    rf = RandomForestClassifier(
        n_estimators=400,
        n_jobs=-1,
        random_state=42
    )

    pipe = Pipeline(steps=[("pre", preproc), ("rf", rf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_pos = f1_score(y_test, y_pred, average="binary", pos_label=1)

    print(f"[metrics] accuracy={acc:.4f} | f1(contratado)= {f1_pos:.4f}")

    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / MODEL_FILENAME
    joblib.dump(pipe, model_path)

    print(f"[preprocess_and_train] Modelo persistido en: {model_path}")
    return str(model_path)


# ---------------------------------------------------------------------
# 4) Inferencia con Gradio (opcional del enunciado)
# ---------------------------------------------------------------------
try:
    import gradio as gr
except Exception:
    gr = None  # Evita fallar si gradio no está instalado en el contenedor


def predict(file, model_path: str):
    pipeline = joblib.load(model_path)
    input_data = pd.read_json(file)
    if TARGET in input_data.columns:
        input_data = input_data.drop(columns=[TARGET])

    preds = pipeline.predict(input_data)
    labels = ["No contratado" if pred == 0 else "Contratado" for pred in preds]

    print(f"[predict] y_hat={preds}")
    return {"Predicción": labels[0]}


def _latest_model_under_runs() -> Optional[str]:
    """Busca el modelo más reciente en dags/runs/*/models/."""
    run_folders = sorted((p for p in RUNS_ROOT.glob("*") if p.is_dir()), key=lambda p: p.name, reverse=True)
    for rd in run_folders:
        cand = rd / "models" / MODEL_FILENAME
        if cand.exists():
            return str(cand)
    return None


def gradio_interface(model_path: Optional[str] = None):
    if gr is None:
        raise RuntimeError("gradio no está instalado en el contenedor. Instálalo si quieres lanzar la UI.")
    model_p = model_path or _latest_model_under_runs()
    if not model_p:
        raise FileNotFoundError("No se encontró un modelo entrenado en 'runs/*/models/'.")
    interface = gr.Interface(
        fn=lambda file: predict(file, model_p),
        inputs=gr.File(label="Sube un archivo JSON"),
        outputs="json",
        title="Hiring Decision Prediction",
        description="Sube un archivo JSON con las características para predecir si será contratada/o."
    )
    interface.launch(share=True)