"""Local test for backend module: load models and call predict() function.

Run from repository root: `python app/backend/test_backend_local.py`
"""
import importlib
import traceback


def main():
    import sys
    from pathlib import Path
    # Ensure repo root is on sys.path so `app` package is importable
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    m = importlib.import_module("app.backend.main")

    # Call startup loader
    try:
        m.load_models()
    except Exception:
        print("load_models raised an exception:")
        traceback.print_exc()

    print("model loaded:", m.model is not None, "using_pipeline", m.using_pipeline)
    print("expected_columns (first 20):", None if m.expected_columns is None else m.expected_columns[:20])

    from app.backend.main import PredictRequest

    # Build a sample instance using expected columns if available
    if m.expected_columns:
        sample = {c: 0 for c in m.expected_columns}
    else:
        sample = {"customer_id": 25734, "product_id": 39186, "week": "2025-01-07"}

    req = PredictRequest(instances=[sample])
    try:
        out = m.predict(req)
        print("predict output:", out)
    except Exception:
        print("predict raised an exception:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
