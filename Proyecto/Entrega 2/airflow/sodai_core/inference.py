# airflow/sodai_core/inference.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import data_io
from . import features as feats
import joblib
from pathlib import Path
from typing import Optional
import mlflow
import json
import logging
try:
    import shap
except Exception:
    shap = None


def _ensure_sklearn_compat_shim() -> None:
    """
    Monkeypatch sklearn internals to improve unpickling compatibility when
    artifacts were created with an older scikit-learn version.

    Some sklearn internals (eg. `_RemainderColsList`) moved between
    versions; unpickling a ColumnTransformer saved with an older
    sklearn can raise AttributeError. Define a minimal compatibility
    placeholder class in the module so pickle can find the symbol.
    """
    try:
        import importlib

        mod = importlib.import_module("sklearn.compose._column_transformer")
        if not hasattr(mod, "_RemainderColsList"):
            # Minimal compatibility placeholder — a simple list subclass
            # matches the original name and allows pickle to construct
            # objects that expect this type. This is safer than failing
            # to load the whole artifact; downstream code should still
            # validate the preprocessor before use.
            class _RemainderColsList(list):
                pass

            setattr(mod, "_RemainderColsList", _RemainderColsList)
    except Exception:
        # If anything goes wrong here, we don't want to break the
        # normal execution path; the loader will raise on joblib.load.
        return


def safe_joblib_load(path: str | Path):
    """Load a joblib file after applying compatibility shims."""
    _ensure_sklearn_compat_shim()
    return joblib.load(path)


# ======================
# Carga de modelo productivo
# ======================

def load_production_model(
    artifacts_dir: Path | str | None = None,
):
    """
    Carga el preprocesador y el modelo considerados "productivos".

    Por defecto utiliza los artefactos de Optuna de la entrega 1,
    pero el DAG puede pasar otra ruta si se reentrena y se guarda una nueva versión.
    """
    if artifacts_dir is None:
        artifacts_dir = data_io.ARTIFACTS_OPTUNA_DIR

    artifacts_dir = Path(artifacts_dir)

    # Prefer direct pipeline if exists (evita problemas con nombres de columnas)
    pipeline_path = artifacts_dir / "pipeline_best.joblib"
    if pipeline_path.exists():
        try:
            pipeline = safe_joblib_load(pipeline_path)
            # Check whether the pipeline's model is fitted; if not, prefer separate artifacts
            try:
                from sklearn.utils.validation import check_is_fitted

                model_step = None
                try:
                    model_step = pipeline.named_steps.get("model")
                except Exception:
                    model_step = None

                if model_step is not None:
                    try:
                        check_is_fitted(model_step)
                    except Exception:
                        logging.warning("Loaded pipeline exists but internal model is not fitted; falling back to separate artifacts")
                        # Immediately return separate artifacts instead of the broken pipeline
                        return data_io.load_best_artifacts(artifacts_dir)
            except Exception:
                # If sklearn isn't available or check failed, continue and let later code handle failures
                pass
            metadata_path = artifacts_dir / "metadata.json"
            metadata = None
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                except Exception:
                    metadata = None
            # devolver pipeline como `model` y `preprocess` vacío para indicar
            # que hay un pipeline que ya contiene el preprocesador
            return None, pipeline, metadata
        except Exception:
            # fallback a la carga clásica
            pass

    preprocess, model, metadata = data_io.load_best_artifacts(artifacts_dir)
    return preprocess, model, metadata


# ======================
# Preparación de candidatos para la próxima semana
# ======================

def build_scoring_candidates(
    interacciones_semana: pd.DataFrame,
    reference_week: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Prepara las filas que se van a puntuar para la "próxima semana".

    Estrategia razonable:
    - Para cada (customer_id, product_id), tomar la fila más reciente disponible
      (última semana observada).
    - Usar sus features como representación del estado actual.
    - La "próxima semana" será, conceptualmente, la siguiente a la semana máxima
      en el dataset.

    Parameters
    ----------
    interacciones_semana:
        DataFrame con al menos las columnas de features ya construidas.
    reference_week:
        Semana de referencia. Si es None, se usa la semana máxima presente
        en el dataset.

    Returns
    -------
    candidates : pd.DataFrame
        Una fila por (customer_id, product_id) lista para pasar por preprocess+modelo.
    """
    df = interacciones_semana.copy()
    if reference_week is None:
        reference_week = df["week"].max()

    # Nos quedamos solo con las semanas <= reference_week (por si en el futuro
    # recibimos datos de más adelante)
    df = df[df["week"] <= reference_week]

    # Tomamos la última fila por (customer_id, product_id)
    df = df.sort_values(["customer_id", "product_id", "week"])
    last = df.groupby(["customer_id", "product_id"], as_index=False).tail(1)

    return last.reset_index(drop=True)


# ======================
# Scoring y ranking
# ======================

def score_candidates(
    candidates: pd.DataFrame,
    preprocess,
    model,
) -> pd.DataFrame:
    """
    Aplica preprocess + modelo a los candidatos y devuelve un DataFrame
    con las probabilidades de compra.

    Se asume que `candidates` ya tiene todas las columnas de features
    que espera el preprocesador (NUM_FEATURES + CAT_FEATURES).
    """
    X = candidates[feats.NUM_FEATURES + feats.CAT_FEATURES].copy()

    # Coerce numeric features to numeric dtype and categorical features to object
    # to minimize dtype mismatches when unpickling preprocessors from different
    # scikit-learn versions. This helps OneHotEncoder and ColumnTransformer
    # avoid internal TypeErrors when comparing categories.
    for col in feats.NUM_FEATURES:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in feats.CAT_FEATURES:
        if col in X.columns:
            # ensure missing values are represented as np.nan and dtype is object
            X[col] = X[col].where(pd.notnull(X[col]), np.nan).astype(object)

    # Si `model` es un Pipeline que ya contiene el preprocesador, usarlo directamente
    try:
        from sklearn.pipeline import Pipeline as SklearnPipeline

        is_pipeline = isinstance(model, SklearnPipeline)
    except Exception:
        is_pipeline = False

    if preprocess is None and is_pipeline:
        # El pipeline interno se encargará del preprocesado; pasar DataFrame original
        proba = model.predict_proba(X)[:, 1]
    else:
        # Aplicar preprocesador
        X_trans = preprocess.transform(X)

        # Intentar pasar un DataFrame con nombres de columnas transformadas al modelo
        # para evitar warnings del tipo "X does not have valid feature names"
        try:
            feat_names = list(preprocess.get_feature_names_out())
            X_for_model = pd.DataFrame(X_trans, columns=feat_names, index=X.index)
            proba = model.predict_proba(X_for_model)[:, 1]
        except Exception:
            # Fallback: pasar array numpy
            proba = model.predict_proba(X_trans)[:, 1]

    out = candidates[feats.ID_COLS].copy()
    out["proba_buy_next_week"] = proba.astype("float32")

    return out


def build_rankings(
    scored: pd.DataFrame,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    Construye el ranking de productos por cliente a partir de las probabilidades.

    Parameters
    ----------
    scored:
        DataFrame con columnas [customer_id, product_id, week, proba_buy_next_week].
    top_k:
        Número de productos máximo a recomendar por cliente.

    Returns
    -------
    ranking : pd.DataFrame
        DataFrame con columnas:
          - customer_id
          - product_id
          - week
          - proba_buy_next_week
          - rank (1 = más probable)
    """
    df = scored.copy()
    df = df.sort_values(["customer_id", "proba_buy_next_week"], ascending=[True, False])

    # Ranking por cliente
    df["rank"] = (
        df.groupby("customer_id")["proba_buy_next_week"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    df = df[df["rank"] <= top_k]

    return df.reset_index(drop=True)


# ======================
# Función de alto nivel para el DAG
# ======================

def run_scoring_pipeline(
    data_dir: Path | str | None = None,
    artifacts_dir: Path | str | None = None,
    reference_week: Optional[pd.Timestamp] = None,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    Función de alto nivel para el DAG:

    - Carga datos crudos.
    - Reconstruye interacciones semanales + features (o las carga si ya existen).
    - Carga modelo productivo.
    - Construye candidatos y los puntúa.
    - Devuelve ranking cliente-producto para la próxima semana.

    En un escenario real, lo ideal es que el DAG:
      - tenga una task que genere/actualice `interacciones_semana`,
      - otra task que reentrene si corresponde,
      - y esta task de scoring que consuma los artefactos más recientes.
    """
    if data_dir is None:
        data_dir = data_io.DATA_DIR
    if artifacts_dir is None:
        artifacts_dir = data_io.ARTIFACTS_OPTUNA_DIR

    # En la entrega 2 se debería decidir:
    #  - o bien regenerar interacciones_semana desde crudo,
    #  - o bien cargarlas si ya existen en data/.
    transacciones, clientes, productos = data_io.load_raw_tables(data_dir)
    transacciones_limpio = feats.clean_transactions(transacciones)
    interacciones = feats.build_weekly_interactions(
        transacciones_limpio, clientes, productos
    )
    interacciones = feats.add_behavioral_features(interacciones)

    preprocess, model, metadata = load_production_model(artifacts_dir)

    candidates = build_scoring_candidates(interacciones, reference_week=reference_week)
    scored = score_candidates(candidates, preprocess, model)
    ranking = build_rankings(scored, top_k=top_k)

    return ranking


def compute_and_log_shap_global(
    *,
    data_dir: Path | str,
    model_path: Optional[str] = None,
    preprocess_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    mlflow_experiment: Optional[str] = None,
    sample_n: int = 1000,
) -> None:
    """
    Calcula un resumen global de SHAP (o un fallback) y lo registra como artefacto.

    - Si `model_path`/`preprocess_path` están definidos, los carga desde disco.
    - Si no, intenta cargar los artefactos productivos por defecto.
    - Guarda un JSON con la importancia media absoluta por feature y lo sube a MLflow
      si `mlflow_experiment` está definido.
    """
    # Cargar artefactos
    if model_path is None or preprocess_path is None:
        preprocess, model, metadata = load_production_model()
    else:
        preprocess = safe_joblib_load(preprocess_path)
        model = safe_joblib_load(model_path)
        metadata = None
        if metadata_path is not None:
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            except Exception:
                metadata = None

    # Cargar/crear dataframe de features
    data_dir = Path(data_dir)
    interacciones_path = data_dir / "interacciones_semana.parquet"
    if interacciones_path.exists():
        df = pd.read_parquet(interacciones_path)
    else:
        # Intentamos regenerar la matriz de features en memoria
        tmp = data_dir / "tmp_feature_matrix_for_shap.parquet"
        try:
            feats.build_feature_matrix_from_parquets(data_dir=data_dir, output_path=tmp)
            df = pd.read_parquet(tmp)
            try:
                tmp.unlink()
            except Exception:
                pass
        except Exception:
            # No podemos calcular SHAP sin datos
            raise RuntimeError("No fue posible obtener la matriz de features para calcular SHAP")

    cols = feats.NUM_FEATURES + feats.CAT_FEATURES
    X = df[cols].copy()
    if len(X) > sample_n:
        Xs = X.sample(sample_n, random_state=42)
    else:
        Xs = X

    shap_summary = {}
    if shap is not None:
        try:
            # Aplicar preprocesador y explicar sobre la representación que recibe el modelo
            # Si no tenemos `preprocess` pero `model` es un Pipeline guardado, extraemos componentes
            try:
                from sklearn.pipeline import Pipeline as SklearnPipeline
            except Exception:
                SklearnPipeline = None

            model_for_explainer = model
            if preprocess is None and SklearnPipeline is not None and isinstance(model, SklearnPipeline):
                # pipeline contains preprocess + model
                try:
                    preprocess = model.named_steps.get("preprocess")
                    model_for_explainer = model.named_steps.get("model")
                except Exception:
                    preprocess = None

            if preprocess is None:
                # No podemos transformar las features; lanzamos para que caiga al fallback
                raise RuntimeError("Preprocessor unavailable for SHAP computation")

            X_trans = preprocess.transform(Xs)
            explainer = shap.TreeExplainer(model_for_explainer)
            # shap_values puede ser grande; calculamos mean(|shap|)
            sv = explainer.shap_values(X_trans)
            # para clasificación binaria, TreeExplainer devuelve lista [neg,pos] o array; intentamos manejar ambos
            if isinstance(sv, list):
                arr = np.abs(sv[1])
            else:
                arr = np.abs(sv)
            mean_abs = np.mean(arr, axis=0)

            # Obtener nombres de columnas después del preprocesador si es ColumnTransformer
            try:
                # Si el preprocessor es ColumnTransformer con get_feature_names_out
                feature_names = list(preprocess.get_feature_names_out(cols))
            except Exception:
                # fallback: usar input feature names (aunque no representen columnas transformadas)
                feature_names = cols

            shap_summary = {k: float(v) for k, v in zip(feature_names, mean_abs)}
        except Exception:
            shap_summary = {
                "error": "Error calculando SHAP; se generó fallback de feature_importances"
            }

    if not shap_summary:
        # Fallback: usar importancias del modelo si están disponibles
        try:
            fi = getattr(model, "feature_importances_", None)
            if fi is not None:
                shap_summary = {f: float(v) for f, v in zip(cols, fi)}
            else:
                shap_summary = {"warning": "No se pudo obtener SHAP ni feature_importances"}
        except Exception:
            shap_summary = {"warning": "No se pudo obtener SHAP ni feature_importances"}

    # Guardar resumen localmente
    out_dir = data_dir / "artifacts_shap"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "shap_summary.json"
    with open(out_path, "w") as f:
        json.dump(shap_summary, f, indent=2)

    # Log a MLflow si corresponde
    if mlflow_experiment is not None:
        mlflow.set_tracking_uri(f"file://{(Path.cwd() / 'mlruns')}" )
        mlflow.set_experiment(mlflow_experiment)
        with mlflow.start_run(run_name="compute_shap_global"):
            mlflow.log_artifact(str(out_path))


def generate_weekly_predictions(
    *,
    data_dir: Path | str,
    model_path: Optional[str] = None,
    preprocess_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    output_dir: str | Path | None = None,
    mlflow_experiment: Optional[str] = None,
    top_k: int = 20,
) -> str:
    """
    Genera el ranking de predicciones para la próxima semana y guarda el parquet.

    Devuelve la ruta al archivo guardado.
    """
    data_dir = Path(data_dir)
    # 1) Reconstruir interacciones y features (similar a run_scoring_pipeline)
    transacciones, clientes, productos = data_io.load_raw_tables(data_dir)
    transacciones_limpio = feats.clean_transactions(transacciones)
    interacciones = feats.build_weekly_interactions(
        transacciones_limpio, clientes, productos
    )
    interacciones = feats.add_behavioral_features(interacciones)

    # 2) Cargar modelo y preprocesador
    if model_path is None or preprocess_path is None:
        preprocess, model, metadata = load_production_model()
    else:
        preprocess = safe_joblib_load(preprocess_path)
        model = safe_joblib_load(model_path)

    # 3) Construir candidatos y puntuar
    candidates = build_scoring_candidates(interacciones)
    scored = score_candidates(candidates, preprocess, model)
    ranking = build_rankings(scored, top_k=top_k)

    # 4) Guardar resultados
    if output_dir is None:
        output_dir = data_dir / "predictions"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    ranking.to_parquet(out_path, index=False)

    # 5) Log en MLflow si corresponde
    if mlflow_experiment is not None:
        mlflow.set_tracking_uri(f"file://{(Path.cwd() / 'mlruns')}")
        mlflow.set_experiment(mlflow_experiment)
        with mlflow.start_run(run_name="generate_weekly_predictions"):
            mlflow.log_artifact(str(out_path))

    return str(out_path)