from datetime import datetime, date
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule

from hiring_dynamic_functions import (
    create_folders,
    load_ands_merge,   # alias load_and_merge está dentro del módulo
    split_data,
    train_model,
    evaluate_models,
)

# ----------------------- Config -----------------------
DATA1_URL = "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
DATA2_URL = "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_2.csv"
CUTOFF = date(2024, 11, 1)              # desde esta fecha (incl.) usar también data_2.csv
SCHEDULE = "0 15 5 * *"                 # 5 de cada mes, 15:00 UTC


# --------------------- Branching ----------------------
def decide_downloads(**context):
    """
    - Fechas previas al 01-11-2024:   -> solo 'download_data_1'
    - Desde 01-11-2024 (incluida):   -> 'download_data_1' y 'download_data_2'
    """
    ds = context["ds"]                                # 'YYYY-MM-DD'
    logical_date = datetime.strptime(ds, "%Y-%m-%d").date()
    if logical_date < CUTOFF:
        return "download_data_1"
    return ["download_data_1", "download_data_2"]


# ------------------------ DAG -------------------------
with DAG(
    dag_id="hiring_dynamic",
    description="Pipeline mensual con branching y entrenamiento paralelo",
    start_date=datetime(2024, 10, 1),    # backfill desde octubre 2024
    schedule=SCHEDULE,
    catchup=True,                        # habilita backfill
    tags=["lab9", "hiring", "dynamic", "parallel"],
    default_args={
        "owner": "airflow",
        "retries": 2,
        "retry_delay": 300,
    },
    max_active_runs=2,
) as dag:

    # marcador de inicio
    start = EmptyOperator(task_id="start")

    # crear carpetas
    mk_dirs = PythonOperator(
        task_id="create_folders",
        python_callable=create_folders,
    )

    # branching
    branch = BranchPythonOperator(
        task_id="branch_downloads",
        python_callable=decide_downloads,
    )

    # descargas
    download1 = BashOperator(
        task_id="download_data_1",
        bash_command=r"""
set -euo pipefail
RAW="{{ ti.xcom_pull(task_ids='create_folders') }}/raw"
mkdir -p "$RAW"
OUT="$RAW/data_1.csv"
echo "[download_data_1] -> $OUT"
curl -fSL --retry 5 --retry-all-errors --retry-delay 5 --connect-timeout 15 --max-time 180 \
  -o "$OUT" "{{ params.url }}"
test -s "$OUT"
""",
        params={"url": DATA1_URL},
    )

    download2 = BashOperator(
        task_id="download_data_2",
        bash_command=r"""
set -euo pipefail
RAW="{{ ti.xcom_pull(task_ids='create_folders') }}/raw"
mkdir -p "$RAW"
OUT="$RAW/data_2.csv"
echo "[download_data_2] -> $OUT"
curl -fSL --retry 5 --retry-all-errors --retry-delay 5 --connect-timeout 15 --max-time 180 \
  -o "$OUT" "{{ params.url }}"
test -s "$OUT"
""",
        params={"url": DATA2_URL},
    )

    # merge (con que una descarga sea exitosa, continúa)
    merge = PythonOperator(
        task_id="load_and_merge",
        python_callable=load_ands_merge,
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    # split
    do_split = PythonOperator(
        task_id="split_data",
        python_callable=split_data,
    )

    # entrenamientos en paralelo
    train_rf = PythonOperator(
        task_id="train_random_forest",
        python_callable=train_model,
        op_kwargs={"estimator": "random_forest",
                   "estimator_params": {"n_estimators": 300, "random_state": 42}},
    )
    train_logreg = PythonOperator(
        task_id="train_logreg",
        python_callable=train_model,
        op_kwargs={"estimator": "logreg",
                   "estimator_params": {"max_iter": 500, "solver": "lbfgs"}},
    )
    train_gbc = PythonOperator(
        task_id="train_gradient_boosting",
        python_callable=train_model,
        op_kwargs={"estimator": "gbc",
                   "estimator_params": {"random_state": 42}},
    )

    # evaluación (solo si los 3 entrenamientos terminaron OK)
    select_best = PythonOperator(
        task_id="evaluate_models",
        python_callable=evaluate_models,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    end = EmptyOperator(task_id="end")

    # Orquestación
    start >> mk_dirs >> branch
    branch >> [download1, download2] >> merge
    merge >> do_split >> [train_rf, train_logreg, train_gbc] >> select_best >> end