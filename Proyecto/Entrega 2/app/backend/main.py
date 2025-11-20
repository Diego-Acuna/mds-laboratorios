from typing import List, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
from pathlib import Path

app = FastAPI(title="Sodai Predictor Backend")

# Candidate artifact locations (container path + repo-relative paths for local dev)
# Compute repo root robustly: in some execution environments (e.g. when the
# app is copied into a shallow container path) `parents[2]` may be out of range.
try:
    REPO_ROOT = Path(__file__).resolve().parents[2]
except Exception:
    REPO_ROOT = Path.cwd()
MODEL_DIR_CANDIDATES = [
    Path("/models"),
    REPO_ROOT / "airflow" / "artifacts_entrega1" / "artifacts_optuna",
    REPO_ROOT / "airflow" / "artifacts_entrega1" / "artifacts",
]

MODEL_PATHS = [
    "pipeline_best.joblib",
    "pipeline_best.pkl",
    "pipeline.joblib",
    "model_lgbm_best.joblib",
    "model_lgbm_v1.joblib",
]
PREPROCESS_PATHS = [
    "preprocess_best.joblib",
    "preprocess.joblib",
    "preprocess_v1.joblib",
]

model = None
preprocessor = None
using_pipeline = False
expected_columns: Optional[List[str]] = None


class PredictRequest(BaseModel):
    instances: List[Any]


def _ensure_sklearn_compat_shim():
    try:
        import importlib

        mod = importlib.import_module("sklearn.compose._column_transformer")
        if not hasattr(mod, "_RemainderColsList"):
            class _RemainderColsList(list):
                pass

            setattr(mod, "_RemainderColsList", _RemainderColsList)
    except Exception:
        return


def safe_joblib_load(path: str):
    _ensure_sklearn_compat_shim()
    return joblib.load(path)


def _find_artifact_path(filename: str) -> Optional[Path]:
    """Search candidate model dirs for filename and return first match."""
    for d in MODEL_DIR_CANDIDATES:
        try:
            p = Path(d) / filename
            if p.exists():
                return p
        except Exception:
            continue
    return None


@app.on_event("startup")
def load_models():
    """Attempt to load a pipeline or (preprocessor + model) from known locations.

    The function tolerates running inside Docker (where artifacts are mounted to
    `/models`) and local development where artifacts are under `airflow/artifacts_entrega1/...`.
    """
    global model, preprocessor, using_pipeline, expected_columns

    # Try pipeline first
    for fname in MODEL_PATHS:
        p = _find_artifact_path(fname)
        if p is None:
            continue
        try:
            loaded = safe_joblib_load(str(p))
            # Distinguish between a full sklearn Pipeline (with preprocess step)
            # and a bare model object that also implements predict_proba.
            try:
                from sklearn.pipeline import Pipeline as SklearnPipeline
            except Exception:
                SklearnPipeline = None

            is_pipeline = SklearnPipeline is not None and isinstance(loaded, SklearnPipeline)

            if is_pipeline:
                model = loaded
                using_pipeline = True
                print(f"Loaded sklearn Pipeline artifact from {p}")
                # Try to infer expected columns from internal preprocess step if present
                try:
                    preprocess = getattr(model, "named_steps", {}).get("preprocess")
                    if preprocess is not None and hasattr(preprocess, "feature_names_in_"):
                        expected_columns = list(preprocess.feature_names_in_)
                except Exception:
                    pass
                return
            else:
                # It's not a Pipeline; might be a bare model (LGBM) — don't assume pipeline
                # Keep it in `model` and continue to try loading a standalone preprocessor.
                model = loaded
                using_pipeline = False
                print(f"Loaded model artifact (non-pipeline) from {p}")
                # Do not return yet; allow the subsequent preprocessor-loading loop to run
                break
        except Exception as e:
            print(f"Failed to load pipeline artifact {p}: {e}")

    # Try to load standalone model and preprocessor
    for fname in MODEL_PATHS:
        p = _find_artifact_path(fname)
        if p is None:
            continue
        try:
            loaded = safe_joblib_load(str(p))
            model = loaded
            print(f"Loaded model artifact from {p}")
            break
        except Exception as e:
            print(f"Failed to load model artifact {p}: {e}")

    for fname in PREPROCESS_PATHS:
        p = _find_artifact_path(fname)
        if p is None:
            continue
        try:
            preprocessor = safe_joblib_load(str(p))
            print(f"Loaded preprocessor artifact from {p}")
            # Try to infer expected input columns
            try:
                if hasattr(preprocessor, "feature_names_in_"):
                    expected_columns = list(preprocessor.feature_names_in_)
                else:
                    # ColumnTransformer fallback
                    from sklearn.compose import ColumnTransformer

                    if isinstance(preprocessor, ColumnTransformer):
                        cols: list = []
                        for name, trans, cols_spec in preprocessor.transformers_:
                            if isinstance(cols_spec, (list, tuple)):
                                for c in cols_spec:
                                    if isinstance(c, str):
                                        cols.append(c)
                        if cols:
                            expected_columns = cols
            except Exception:
                expected_columns = None
            break
        except Exception as e:
            print(f"Failed to load preprocessor artifact {p}: {e}")

    if model is None:
        print("No model artifact found in candidate locations. Place model files under /models or airflow/artifacts_entrega1/... to enable /predict.")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server")
    instances = request.instances
    if not isinstance(instances, list) or len(instances) == 0:
        raise HTTPException(status_code=400, detail="`instances` must be a non-empty list of records")
    # Normalize into DataFrame
    try:
        if isinstance(instances[0], dict):
            df = pd.DataFrame(instances)
        else:
            # list of lists: user must provide column names via first row or use dicts
            df = pd.DataFrame(instances)
        # If we know expected columns, ensure dataframe has those columns in order,
        # filling missing columns with NaN and ignoring unexpected ones.
        if expected_columns is not None:
            n = len(df)
            df_full = pd.DataFrame(index=range(n), columns=expected_columns)
            for c in df.columns:
                if c in expected_columns:
                    df_full[c] = df[c].values
            df = df_full
        # Coerce object dtypes: if a column is fully numeric-like, convert to numeric.
        # Otherwise keep as object (strings) and normalize missing values to np.nan.
        for col in df.columns:
            if df[col].dtype == object:
                coerced = pd.to_numeric(df[col], errors='coerce')
                if coerced.notnull().all():
                    df[col] = coerced
                else:
                    df[col] = df[col].where(pd.notnull(df[col]), np.nan).astype(object)
        # Convert boolean columns to int
        for col in df.columns:
            if df[col].dtype.name == 'bool':
                df[col] = df[col].astype(int)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse instances into DataFrame: {e}")
    try:
        if using_pipeline:
            # The loaded 'model' is a pipeline with predict_proba
            probs = model.predict_proba(df)
            # Support binary classification returning 2 cols
            if probs.ndim == 1 or probs.shape[1] == 1:
                result = [float(p) for p in probs]
            else:
                result = [float(p[1]) for p in probs]
        else:
            # Need preprocessor + model
            if preprocessor is None:
                raise HTTPException(status_code=500, detail="Preprocessor missing for non-pipeline model")
            # Coerce columns according to preprocessor's declared transformers
            try:
                num_cols = []
                cat_cols = []
                if hasattr(preprocessor, "transformers_"):
                    for name, trans, cols_spec in preprocessor.transformers_:
                        if not isinstance(cols_spec, (list, tuple)):
                            continue
                        cols_list = [c for c in cols_spec if isinstance(c, str)]
                        if "num" in name:
                            num_cols.extend(cols_list)
                        elif "cat" in name:
                            cat_cols.extend(cols_list)

                for c in num_cols:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")

                for c in cat_cols:
                    if c in df.columns:
                        df[c] = df[c].where(pd.notnull(df[c]), np.nan).astype(object)
            except Exception:
                pass

            X = preprocessor.transform(df)
            probs = model.predict_proba(X)
            if probs.ndim == 1 or probs.shape[1] == 1:
                result = [float(p) for p in probs]
            else:
                result = [float(p[1]) for p in probs]
        return {"predictions": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
