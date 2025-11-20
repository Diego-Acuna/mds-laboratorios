# test_inference.py
# Run inside the airflow-scheduler container:
# docker compose exec airflow-scheduler python /opt/airflow/scripts/test_inference.py

import joblib
import importlib


def _ensure_sklearn_compat_shim():
    try:
        mod = importlib.import_module("sklearn.compose._column_transformer")
        if not hasattr(mod, "_RemainderColsList"):
            class _RemainderColsList(list):
                pass

            setattr(mod, "_RemainderColsList", _RemainderColsList)
    except Exception:
        pass


def safe_joblib_load(p):
    _ensure_sklearn_compat_shim()
    return joblib.load(p)
import pyarrow.parquet as pq
from pathlib import Path

PIPE = Path('/opt/airflow/models/pipeline_best.joblib')
PP = Path('/opt/airflow/models/preprocess_best.joblib')
MDL = Path('/opt/airflow/models/model_lgbm_best.joblib')
FEATURE_MATRIX = Path('/opt/airflow/data/working/feature_matrix_latest.parquet')

print('Checking files:')
print(' pipeline:', PIPE.exists(), PIPE)
print(' preprocess:', PP.exists(), PP)
print(' model:', MDL.exists(), MDL)
print(' feature matrix:', FEATURE_MATRIX.exists(), FEATURE_MATRIX)

if not FEATURE_MATRIX.exists():
    raise SystemExit('feature matrix not found')

# load a tiny sample
tbl = pq.read_table(FEATURE_MATRIX)
df = tbl.to_pandas()
sample = df.head(5).drop(columns=['bought'], errors='ignore')
print('sample shape', sample.shape)

# try pipeline
if PIPE.exists():
    try:
        pipe = safe_joblib_load(PIPE)
        print('Loaded pipeline from', PIPE)
        try:
            probs = pipe.predict_proba(sample)[:, 1]
            print('pipeline.predict_proba ->', probs)
        except Exception as e:
            print('pipeline.predict_proba failed:', type(e).__name__, e)
    except Exception as e:
        print('Failed loading pipeline:', type(e).__name__, e)

# fallback to preprocess + model
if PP.exists() and MDL.exists():
    try:
        pp = safe_joblib_load(PP)
        mdl = safe_joblib_load(MDL)
        print('Loaded preprocess and model separately')
        X = pp.transform(sample)
        print('Transformed shape:', getattr(X, 'shape', None))
        probs2 = mdl.predict_proba(X)[:, 1]
        print('model.predict_proba ->', probs2)
    except Exception as e:
        print('Fallback error:', type(e).__name__, e)
else:
    print('No separate preprocess/model artifacts available to try fallback')
