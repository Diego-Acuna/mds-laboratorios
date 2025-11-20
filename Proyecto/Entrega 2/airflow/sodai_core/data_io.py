# airflow/sodai_core/data_io.py

from __future__ import annotations

from pathlib import Path
import json
from typing import Tuple, Dict, Any

import pandas as pd
import joblib
import logging
from sklearn.pipeline import Pipeline


def _ensure_sklearn_compat_shim() -> None:
    """
    Monkeypatch sklearn internals to improve unpickling compatibility when
    artifacts were created with an older scikit-learn version.

    This mirrors the shim in `inference.py` and is defensive when loading
    artifacts from disk inside the Airflow tasks.
    """
    try:
        import importlib

        mod = importlib.import_module("sklearn.compose._column_transformer")
        if not hasattr(mod, "_RemainderColsList"):
            class _RemainderColsList(list):
                pass

            setattr(mod, "_RemainderColsList", _RemainderColsList)
    except Exception:
        return


def safe_joblib_load(path):
    _ensure_sklearn_compat_shim()
    return joblib.load(path)


# Raíz del proyecto de Airflow (carpeta "airflow")
AIRFLOW_ROOT = Path(__file__).resolve().parents[1]

# Directorios por defecto
DATA_DIR = AIRFLOW_ROOT / "data"
ARTIFACTS_ENTREGA1_DIR = AIRFLOW_ROOT / "artifacts_entrega1"
ARTIFACTS_V1_DIR = ARTIFACTS_ENTREGA1_DIR / "artifacts"
ARTIFACTS_OPTUNA_DIR = ARTIFACTS_ENTREGA1_DIR / "artifacts_optuna"


# ======================
# Carga de datos crudos
# ======================

def load_raw_tables(
    data_dir: Path | str = DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga tablas crudas de transacciones, clientes y productos.

    Esta función debe usarse tanto en notebooks como en DAGs para
    garantizar que todos leen los datos desde el mismo lugar.

    Parameters
    ----------
    data_dir:
        Directorio donde se encuentran los archivos parquet.

    Returns
    -------
    transacciones : pd.DataFrame
    clientes      : pd.DataFrame
    productos     : pd.DataFrame
    """
    data_dir = Path(data_dir)

    transacciones = pd.read_parquet(data_dir / "transacciones.parquet")
    clientes = pd.read_parquet(data_dir / "clientes.parquet")
    productos = pd.read_parquet(data_dir / "productos.parquet")

    return transacciones, clientes, productos

def load_raw_data(
    data_dir: Path | str = DATA_DIR,
):
    """
    Wrapper usado por el DAG de Airflow.

    Mantiene compatibilidad con el nombre `load_raw_data` que aparece
    en `sodai_ml_pipeline_dag.py`, pero internamente reutiliza
    la función `load_raw_tables`.
    """
    return load_raw_tables(data_dir)


def save_weekly_interactions(
    interacciones_semana: pd.DataFrame,
    path: Path | str | None = None,
) -> Path:
    """
    Guarda la tabla semanal de interacciones cliente-producto.

    En la entrega 1 esta tabla se generó en el notebook.
    Aquí centralizamos el guardado para reutilizarla en el DAG.

    Parameters
    ----------
    interacciones_semana:
        DataFrame con una fila por (customer_id, product_id, semana).
    path:
        Ruta opcional de salida. Si es None, se usa data/interacciones_semana.parquet.

    Returns
    -------
    path_out : Path
        Ruta final donde fue guardado el archivo.
    """
    if path is None:
        path = DATA_DIR / "interacciones_semana.parquet"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    interacciones_semana.to_parquet(path, index=False)
    return path


def load_weekly_interactions(path: Path | str | None = None) -> pd.DataFrame:
    """
    Carga la tabla semanal de interacciones cliente-producto.

    Se asume que fue generada en la entrega 1 o por el pipeline de Airflow.

    Parameters
    ----------
    path:
        Ruta del parquet. Si es None, se usa data/interacciones_semana.parquet.

    Returns
    -------
    interacciones_semana : pd.DataFrame
    """
    if path is None:
        path = DATA_DIR / "interacciones_semana.parquet"
    path = Path(path)
    return pd.read_parquet(path)


# ======================
# Carga / guardado de artefactos de modelos
# ======================

def load_v1_artifacts(
    artifacts_dir: Path | str = ARTIFACTS_V1_DIR,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Carga los artefactos "v1" de la entrega 1:
    - preprocess_v1.joblib
    - model_lgbm_v1.joblib
    - metadata_lgbm_v1.json

    Esta versión corresponde al primer modelo antes de tuning.
    """
    artifacts_dir = Path(artifacts_dir)

    preprocess = safe_joblib_load(artifacts_dir / "preprocess_v1.joblib")
    model = safe_joblib_load(artifacts_dir / "model_lgbm_v1.joblib")

    with open(artifacts_dir / "metadata_lgbm_v1.json", "r") as f:
        metadata = json.load(f)

    return preprocess, model, metadata


def load_best_artifacts(
    artifacts_dir: Path | str = ARTIFACTS_OPTUNA_DIR,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Carga los artefactos "best" de la entrega 1 (Optuna):
    - preprocess_best.joblib
    - model_lgbm_best.joblib
    - metadata.json

    Estos son los artefactos que se usarán como modelo productivo.
    """
    artifacts_dir = Path(artifacts_dir)

    # Prefer a saved pipeline (preprocess + model) if exists
    pipeline_path = artifacts_dir / "pipeline_best.joblib"
    if pipeline_path.exists():
        try:
            pipeline = safe_joblib_load(pipeline_path)
            # Extraer componentes si es un sklearn Pipeline
            try:
                preprocess = pipeline.named_steps.get("preprocess")
                model = pipeline.named_steps.get("model")
            except Exception:
                # fallback: keep pipeline as `model`
                preprocess = None
                model = pipeline
        except Exception:
            # Si no podemos cargar el pipeline combinado, intentamos cargar archivos separados
            preprocess = None
            model = None
    else:
        preprocess = None
        model = None

    # Si no cargamos pipeline combinado, intentamos cargar los componentes por separado
    if model is None or preprocess is None:
        # Cargar preprocess si existe
        try:
            preprocess_path = artifacts_dir / "preprocess_best.joblib"
            if preprocess_path.exists():
                try:
                    preprocess = safe_joblib_load(preprocess_path)
                except Exception as e:
                    logging.warning(
                        "Could not load preprocess from %s: %s", preprocess_path, repr(e)
                    )
                    # No queremos que un error de unpickle detenga todo el DAG
                    preprocess = None
        except Exception as e:
            logging.warning("Error checking preprocess path %s: %s", artifacts_dir, repr(e))
            preprocess = None

        # Cargar modelo
        try:
            model_path = artifacts_dir / "model_lgbm_best.joblib"
            if model_path.exists():
                try:
                    model = safe_joblib_load(model_path)
                except Exception as e:
                    logging.warning("Could not load model from %s: %s", model_path, repr(e))
                    model = None
        except Exception as e:
            logging.warning("Error checking model path %s: %s", artifacts_dir, repr(e))
            model = None

    with open(artifacts_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    return preprocess, model, metadata


def save_model_artifacts(
    preprocess,
    model,
    metadata: Dict[str, Any],
    out_dir: Path | str,
    prefix: str = "best",
) -> None:
    """
    Guarda preprocesador, modelo y metadatos en un directorio.

    Esta función puede usarse tanto para reemplazar los artefactos de producción
    como para generar una nueva versión (por ejemplo, al reentrenar con datos nuevos).

    Parameters
    ----------
    preprocess:
        Objeto sklearn (ColumnTransformer / Pipeline) ya ajustado.
    model:
        Modelo final (por ejemplo, LGBMClassifier) ya entrenado.
    metadata:
        Diccionario con información relevante (métricas, hiperparámetros, etc.).
    out_dir:
        Directorio de salida donde se guardarán los artefactos.
    prefix:
        Prefijo para los nombres de archivo, por ejemplo "best" o "v2".
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocess, out_dir / f"preprocess_{prefix}.joblib")
    logging.info("Saved preprocess to %s", out_dir / f"preprocess_{prefix}.joblib")
    joblib.dump(model, out_dir / f"model_lgbm_{prefix}.joblib")
    logging.info("Saved model to %s", out_dir / f"model_lgbm_{prefix}.joblib")
    # Guardar pipeline combinado (preprocess + model) para facilitar inferencia
    try:
        pipeline = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
        joblib.dump(pipeline, out_dir / f"pipeline_{prefix}.joblib")
        logging.info("Saved combined pipeline to %s", out_dir / f"pipeline_{prefix}.joblib")
    except Exception:
        # No queremos fallar el guardado de artefactos por este motivo
        pass
    # Guardamos metadata tanto con prefijo como con nombre estándar
    meta_prefixed = out_dir / f"metadata_{prefix}.json"
    with open(meta_prefixed, "w") as f:
        json.dump(metadata, f, indent=2)
    logging.info("Saved metadata to %s", meta_prefixed)

    # También escribimos un archivo `metadata.json` (compatibilidad con carga)
    meta_standard = out_dir / "metadata.json"
    try:
        # Escribimos la misma información en metadata.json para facilitar cargas
        with open(meta_standard, "w") as f:
            json.dump(metadata, f, indent=2)
        logging.info("Saved metadata to %s", meta_standard)
    except Exception:
        # No queremos fallar el guardado de artefactos por este motivo
        pass