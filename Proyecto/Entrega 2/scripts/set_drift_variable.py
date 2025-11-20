#!/usr/bin/env python3
"""
Setea la Variable `sodai_drift_threshold` en Airflow via REST API.

Uso:
  python scripts/set_drift_variable.py --value 0.0001

Requiere que la API de Airflow esté disponible y, si corresponde, que se provean
credenciales en las variables de entorno `AIRFLOW_USER` y `AIRFLOW_PASSWORD`.
"""
import os
import argparse
import requests
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--value", required=True, help="Valor para sodai_drift_threshold")
    p.add_argument("--airflow-host", default=os.environ.get("AIRFLOW_HOST", "http://localhost:8080"))
    p.add_argument("--username", default=os.environ.get("AIRFLOW_USER"))
    p.add_argument("--password", default=os.environ.get("AIRFLOW_PASSWORD"))
    args = p.parse_args()

    url = args.airflow_host.rstrip("/") + "/api/v1/variables"
    payload = {"key": "sodai_drift_threshold", "value": str(args.value)}

    auth = None
    if args.username and args.password:
        auth = (args.username, args.password)

    headers = {"Content-Type": "application/json"}
    print(f"POST {url} payload={payload} auth={'yes' if auth else 'no'})")
    r = requests.post(url, auth=auth, headers=headers, data=json.dumps(payload))
    if r.status_code in (200, 201):
        print("Variable creada/actualizada con éxito.")
        return

    # Manejo de errores más amigable
    if r.status_code == 403:
        print("Error 403 Forbidden al intentar crear la variable.")
        print("Esto suele ocurrir cuando el usuario usado para la API no tiene permisos RBAC de Admin.")
        print("Opciones para resolverlo:")
        print("  1) Abrir Airflow UI -> Admin -> Variables y crear 'sodai_drift_threshold' manualmente.")
        print("  2) Crear un usuario Admin dentro del contenedor de Airflow y volver a ejecutar el script:")
        print("     docker compose exec airflow-webserver airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin")
        print("     (ajusta el nombre del servicio si tu compose usa otros nombres)")
        print("  3) Alternativa: usar CLI para setear la variable desde el contenedor:")
        print("     docker compose exec airflow-webserver airflow variables set sodai_drift_threshold 0.0001")
        print("Si quieres, puedo guiarte por cualquiera de estas opciones.")
        print("Detalle de la respuesta del servidor:", r.status_code, r.text)
        r.raise_for_status()

    # Otros errores: mostrar detalle y lanzar excepción
    print("Error seteando variable:", r.status_code, r.text)
    r.raise_for_status()


if __name__ == "__main__":
    main()
