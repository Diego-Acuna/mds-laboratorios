from __future__ import annotations

import os
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
from airflow.utils.trigger_rule import TriggerRule

# Importamos ML de la entrega 1 desde sodai_core
from sodai_core import data_io, features, training, inference


# =====================================================================
# Constantes de paths dentro del contenedor de Airflow
# =====================================================================

AIRFLOW_HOME = Path(os.environ.get("AIRFLOW_HOME", "/opt/airflow"))

DATA_DIR = AIRFLOW_HOME / "data"
WORK_DIR = DATA_DIR / "working"
WORK_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACTS_ENTREGA1_DIR = AIRFLOW_HOME / "artifacts_entrega1"
ARTIFACTS_BASELINE_DIR = ARTIFACTS_ENTREGA1_DIR / "artifacts_optuna"

BASELINE_MODEL_PATH = ARTIFACTS_BASELINE_DIR / "model_lgbm_best.joblib"
BASELINE_PREPROCESS_PATH = ARTIFACTS_BASELINE_DIR / "preprocess_best.joblib"
BASELINE_METADATA_PATH = ARTIFACTS_BASELINE_DIR / "metadata.json"

MATRIX_CACHE_PATH = ARTIFACTS_ENTREGA1_DIR / "artifacts" / "matrix_cache_v1.npz"

MODELS_DIR = AIRFLOW_HOME / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_DIR = DATA_DIR / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_TRACKING_URI_DEFAULT = f"file://{AIRFLOW_HOME / 'mlruns'}"


# =====================================================================
# Funciones Python para los operadores
# =====================================================================

def extract_data_task(**context) -> dict:
    """
    Verifica que existen los .parquet necesarios y retorna el data_dir.
    No carga todo en memoria de forma permanente; el trabajo pesado
    se delega a sodai_core.
    """
    tx_path = DATA_DIR / "transacciones.parquet"
    cli_path = DATA_DIR / "clientes.parquet"
    prod_path = DATA_DIR / "productos.parquet"

    missing = []
    if not tx_path.exists():
        missing.append(str(tx_path))
    if not cli_path.exists():
        missing.append(str(cli_path))
    if not prod_path.exists():
        missing.append(str(prod_path))

    if missing:
        raise FileNotFoundError(
            f"Faltan archivos .parquet requeridos en {DATA_DIR}: {missing}"
        )

    # pequeña validación rápida
    df_tx, df_cli, df_prod = data_io.load_raw_data(DATA_DIR)
    if df_tx.empty:
        raise ValueError("El archivo transacciones.parquet está vacío.")
    if df_cli.empty:
        raise ValueError("El archivo clientes.parquet está vacío.")
    if df_prod.empty:
        raise ValueError("El archivo productos.parquet está vacío.")

    return {"data_dir": str(DATA_DIR)}


def build_features_task(ti, **context) -> dict:
    """
    Construye una matriz de features actualizada a partir de los .parquet
    y la guarda en WORK_DIR. Esta matriz puede ser usada para drift,
    entrenamiento y evaluación.
    """
    meta = ti.xcom_pull(task_ids="extract_data")
    data_dir = Path(meta["data_dir"])

    feature_matrix_path = WORK_DIR / "feature_matrix_latest.parquet"

    # Esta función se implementa como wrapper en sodai_core.features
    # reutilizando la lógica de Feature Engineering de la entrega 1.
    features.build_feature_matrix_from_parquets(
        data_dir=data_dir,
        output_path=feature_matrix_path,
    )

    if not feature_matrix_path.exists():
        raise FileNotFoundError(
            f"No se encontró la matriz de features esperada en {feature_matrix_path}"
        )

    return {
        "feature_matrix_path": str(feature_matrix_path),
    }


def detect_drift_task(ti, **context) -> str:
    """
    Detecta drift sencillo comparando tasa de positivos actual vs histórica.
    Si el cambio relativo supera un umbral, dispara reentrenamiento.
    Devuelve el task_id siguiente: 'retrain_model' o 'skip_retraining'.
    """
    # Umbral configurable vía Variable de Airflow (por defecto 0.1 = 10%)
    try:
        drift_threshold = float(Variable.get("sodai_drift_threshold", 0.1))
    except Exception:
        drift_threshold = 0.1

    # Carga proporción histórica de positivos desde matrix_cache_v1.npz
    if not MATRIX_CACHE_PATH.exists():
        # Si por alguna razón no existe, no hacemos reentrenamiento automático
        return "skip_retraining"

    cache = np.load(MATRIX_CACHE_PATH)
    if "y_train" not in cache:
        return "skip_retraining"

    ref_pos_rate = float(cache["y_train"].mean())

    # Carga proporción actual desde la matriz de features
    feat_meta = ti.xcom_pull(task_ids="build_features")
    feature_matrix_path = Path(feat_meta["feature_matrix_path"])
    df_feat = pd.read_parquet(feature_matrix_path)

    if "bought" not in df_feat.columns:
        # Si la columna objetivo no está disponible aquí,
        # asumimos que no tenemos base comparable y no reentrenamos.
        return "skip_retraining"

    new_pos_rate = float(df_feat["bought"].mean())

    # Cambio relativo
    denom = ref_pos_rate if ref_pos_rate > 1e-6 else 1e-6
    rel_change = abs(new_pos_rate - ref_pos_rate) / denom

    # Se podría loguear esto en MLflow o en logs de Airflow
    print(
        f"[DRIFT] ref_pos_rate={ref_pos_rate:.4f}, "
        f"new_pos_rate={new_pos_rate:.4f}, "
        f"rel_change={rel_change:.4f}, "
        f"threshold={drift_threshold:.4f}"
    )

    if rel_change >= drift_threshold:
        return "retrain_model"
    return "skip_retraining"


def retrain_model_task(ti, **context) -> None:
    """
    Reentrena el modelo usando datos actuales e hiperparámetros (por ejemplo
    los mejores de Optuna) y guarda los artefactos en MODELS_DIR.
    Devuelve, vía XCom, un diccionario con las rutas de artefactos.
    """
    mlflow_tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        MLFLOW_TRACKING_URI_DEFAULT,
    )
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    meta = ti.xcom_pull(task_ids="extract_data")
    data_dir = meta["data_dir"]

    # Permitir forzar reentrenamiento via Variable de Airflow
    try:
        force_retrain = Variable.get("sodai_force_retrain", "false").lower() in (
            "1",
            "true",
            "yes",
        )
    except Exception:
        force_retrain = False

    result = training.retrain_model_with_optuna(
        data_dir=data_dir,
        artifacts_entrega1_dir=str(ARTIFACTS_ENTREGA1_DIR),
        artifacts_output_dir=str(MODELS_DIR),
        mlflow_experiment="sodai_retraining",
        force_retrain=force_retrain,
    )

    # Se espera que 'result' tenga al menos estas claves
    # {
    #   "model_path": "...",
    #   "preprocess_path": "...",
    #   "metadata_path": "...",
    #   ...
    # }

    if "model_path" not in result or "preprocess_path" not in result:
        raise ValueError(
            "training.retrain_model_with_optuna debe devolver al menos "
            "'model_path' y 'preprocess_path' en el diccionario result."
        )

    result["source"] = "retrained"

    ti.xcom_push(key="model_artifacts", value=result)


def skip_retraining_task(ti, **context) -> None:
    """
    Rama que se ejecuta cuando NO hay drift. Simplemente informa que
    se debe usar el modelo baseline de la entrega 1.
    """
    if not BASELINE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo baseline en {BASELINE_MODEL_PATH}"
        )
    if not BASELINE_PREPROCESS_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el preprocesador baseline en {BASELINE_PREPROCESS_PATH}"
        )
    if not BASELINE_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el metadata baseline en {BASELINE_METADATA_PATH}"
        )

    result = {
        "model_path": str(BASELINE_MODEL_PATH),
        "preprocess_path": str(BASELINE_PREPROCESS_PATH),
        "metadata_path": str(BASELINE_METADATA_PATH),
        "source": "baseline",
    }

    ti.xcom_push(key="model_artifacts", value=result)


def compute_shap_and_log_task(ti, **context) -> None:
    """
    Calcula SHAP global para el modelo seleccionado (baseline o reentrenado)
    y lo registra (idealmente en MLflow como artefactos).
    """
    mlflow_tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        MLFLOW_TRACKING_URI_DEFAULT,
    )
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    meta = ti.xcom_pull(task_ids="extract_data")
    data_dir = meta["data_dir"]

    # Primero intentamos tomar artefactos de la rama de reentrenamiento;
    # si no existe, usamos la rama que salta el reentrenamiento.
    artifacts = ti.xcom_pull(task_ids="retrain_model", key="model_artifacts")
    if artifacts is None:
        artifacts = ti.xcom_pull(task_ids="skip_retraining", key="model_artifacts")

    if artifacts is None:
        raise ValueError(
            "No se encontraron artefactos de modelo en XCom. "
            "Verifique las tareas 'retrain_model' y 'skip_retraining'."
        )

    inference.compute_and_log_shap_global(
        data_dir=data_dir,
        model_path=artifacts["model_path"],
        preprocess_path=artifacts["preprocess_path"],
        metadata_path=artifacts["metadata_path"],
        mlflow_experiment="sodai_shap",
    )


def generate_predictions_task(ti, **context) -> dict:
    """
    Genera predicciones para la semana siguiente a la última presente
    en los datos, usando el mejor modelo disponible (baseline o reentrenado).
    """
    mlflow_tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        MLFLOW_TRACKING_URI_DEFAULT,
    )
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    meta = ti.xcom_pull(task_ids="extract_data")
    data_dir = meta["data_dir"]

    artifacts = ti.xcom_pull(task_ids="retrain_model", key="model_artifacts")
    if artifacts is None:
        artifacts = ti.xcom_pull(task_ids="skip_retraining", key="model_artifacts")

    if artifacts is None:
        raise ValueError(
            "No se encontraron artefactos de modelo en XCom para generar predicciones."
        )

    try:
        preds_path = inference.generate_weekly_predictions(
            data_dir=data_dir,
            model_path=artifacts["model_path"],
            preprocess_path=artifacts["preprocess_path"],
            metadata_path=artifacts.get("metadata_path"),
            output_dir=str(PREDICTIONS_DIR),
            mlflow_experiment="sodai_inference",
        )
        print(f"Predicciones generadas en: {preds_path}")
        return {"predictions_path": preds_path}
    except Exception as e:
        # If the chosen artifacts fail (e.g., model not fitted), attempt to fallback
        # to the baseline artifacts from the original entrega1.
        print("Error generating predictions with chosen artifacts:", type(e).__name__, e)
        print("Attempting fallback to baseline artifacts...")

        if not BASELINE_MODEL_PATH.exists() or not BASELINE_PREPROCESS_PATH.exists():
            # Nothing to fallback to — re-raise the original error to mark task failed
            raise

        try:
            preds_path = inference.generate_weekly_predictions(
                data_dir=data_dir,
                model_path=str(BASELINE_MODEL_PATH),
                preprocess_path=str(BASELINE_PREPROCESS_PATH),
                metadata_path=str(BASELINE_METADATA_PATH) if BASELINE_METADATA_PATH.exists() else None,
                output_dir=str(PREDICTIONS_DIR),
                mlflow_experiment="sodai_inference_baseline",
            )
            print(f"Fallback predictions generated in: {preds_path}")
            return {"predictions_path": preds_path}
        except Exception:
            # If fallback also fails, re-raise the original exception for visibility
            raise


# =====================================================================
# Definición del DAG
# =====================================================================

default_args = {
    "owner": "deep_drinkers",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="sodai_ml_pipeline",
    default_args=default_args,
    description=(
        "Pipeline productivo de SodAI Drinks: "
        "extracción -> features -> drift -> (retrain/baseline) -> SHAP -> predicciones."
    ),
    schedule_interval="@weekly",      # ajustable según necesidad
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["sodai", "mlops", "mlflow", "optuna", "shap"],
) as dag:

    start = EmptyOperator(task_id="start")

    extract_data = PythonOperator(
        task_id="extract_data",
        python_callable=extract_data_task,
    )

    build_features = PythonOperator(
        task_id="build_features",
        python_callable=build_features_task,
    )

    detect_drift = BranchPythonOperator(
        task_id="detect_drift",
        python_callable=detect_drift_task,
    )

    retrain_model = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model_task,
    )

    skip_retraining = PythonOperator(
        task_id="skip_retraining",
        python_callable=skip_retraining_task,
    )

    compute_shap_and_log = PythonOperator(
        task_id="compute_shap_and_log",
        python_callable=compute_shap_and_log_task,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    generate_predictions = PythonOperator(
        task_id="generate_predictions",
        python_callable=generate_predictions_task,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    end = EmptyOperator(task_id="end")

    # Definición del grafo
    start >> extract_data >> build_features >> detect_drift
    detect_drift >> retrain_model
    detect_drift >> skip_retraining

    [retrain_model, skip_retraining] >> compute_shap_and_log >> generate_predictions >> end