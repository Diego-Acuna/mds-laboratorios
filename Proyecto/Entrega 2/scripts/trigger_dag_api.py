#!/usr/bin/env python3
"""
Trigger a DAG de Airflow via REST API (opcional, si tu instancia expone la API).

Variables de entorno útiles:
  AIRFLOW_HOST (default http://localhost:8080)
  AIRFLOW_USER / AIRFLOW_PASSWORD (si tu Airflow requiere auth básica)

Ejemplo:
  python scripts/trigger_dag_api.py --dag-id sodai_ml_pipeline
"""
import os
import argparse
import requests
import json
from datetime import datetime


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dag-id", default="sodai_ml_pipeline")
    p.add_argument("--airflow-host", default=os.environ.get("AIRFLOW_HOST", "http://localhost:8080"))
    p.add_argument("--username", default=os.environ.get("AIRFLOW_USER"))
    p.add_argument("--password", default=os.environ.get("AIRFLOW_PASSWORD"))
    args = p.parse_args()

    url = args.airflow_host.rstrip("/") + f"/api/v1/dags/{args.dag_id}/dagRuns"
    payload = {"dag_run_id": f"manual__{datetime.utcnow().isoformat()}"}

    auth = None
    if args.username and args.password:
        auth = (args.username, args.password)

    headers = {"Content-Type": "application/json"}
    print(f"POST {url} payload={payload} auth={'yes' if auth else 'no'})")
    r = requests.post(url, auth=auth, headers=headers, data=json.dumps(payload))
    try:
        r.raise_for_status()
    except Exception as e:
        print("Error al trigger DAG:", r.status_code, r.text)
        raise
    print("DAG triggered successfully:", r.json())


if __name__ == "__main__":
    main()
