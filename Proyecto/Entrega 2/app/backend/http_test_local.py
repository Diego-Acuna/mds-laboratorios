"""HTTP-style test for the FastAPI backend using TestClient.

This runs the app in-process (calls startup events) and posts a sample payload
to `/predict` to validate the full request handling path.
"""
from pathlib import Path
import sys
import json

# Ensure repo root is importable
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from fastapi.testclient import TestClient

from app.backend import main as backend


# Ensure models loaded (call explicitly to make behavior deterministic in tests)
try:
    backend.load_models()
except Exception:
    pass


def main():
    client = TestClient(backend.app)

    print('GET /health')
    r = client.get('/health')
    print(r.status_code, r.json())

    # Build a sample using expected columns if available
    sample = None
    if backend.expected_columns:
        sample = {c: 0 for c in backend.expected_columns}
    else:
        sample = {'customer_id': 25734, 'product_id': 39186, 'week': '2025-01-07'}

    payload = {'instances': [sample]}
    print('POST /predict payload keys:', list(payload.keys()))
    r = client.post('/predict', json=payload)
    print('status', r.status_code)
    try:
        print('response json:', json.dumps(r.json(), indent=2))
    except Exception:
        print('response text:', r.text)


if __name__ == '__main__':
    main()
