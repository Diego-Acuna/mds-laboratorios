from datetime import datetime
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from hiring_functions import create_folders, split_data, preprocess_and_train

DATA_URL = "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"

with DAG(
    dag_id="hiring_lineal",
    start_date=datetime(2024, 10, 1),
    schedule=None,
    catchup=False,
    tags=["lab9", "hiring", "lineal"],
    default_args={"owner": "airflow"},
) as dag:

    start_pipeline = EmptyOperator(task_id="start_pipeline")
    end_pipeline = EmptyOperator(task_id="end_pipeline")

    create_folders_task = PythonOperator(
        task_id="create_folders",
        python_callable=create_folders,
    )

    download_data = BashOperator(
        task_id="download_data",
        bash_command=r"""
set -euo pipefail
RAW_DIR="{{ ti.xcom_pull(task_ids='create_folders') }}/raw"
mkdir -p "$RAW_DIR"
echo "[download_data] guardando en: $RAW_DIR/data_1.csv"

curl -fSL --retry 3 --connect-timeout 10 --max-time 120 \
  -o "$RAW_DIR/data_1.csv" "{{ params.url }}"

test -s "$RAW_DIR/data_1.csv"
echo "[download_data] Tamaño:"; ls -lh "$RAW_DIR/data_1.csv"
echo "[download_data] Preview:"; head -n 3 "$RAW_DIR/data_1.csv"
""",
        params={"url": DATA_URL},
    )

    split_task = PythonOperator(
        task_id="split_data",
        python_callable=split_data,
    )

    train_task = PythonOperator(
        task_id="preprocess_and_train",
        python_callable=preprocess_and_train,
    )

    # Lanzamos Gradio como proceso separado
    gradio_task = BashOperator(
        task_id="gradio_interface",
        bash_command=r"""
set -euo pipefail
PORT="${GRADIO_PORT:-7860}"
LOGF="/opt/airflow/logs/gradio_app.log"
PIDF="/opt/airflow/logs/gradio_app.pid"

# Si ya hay uno, lo dejamos
echo "[gradio] Lanzando en puerto ${PORT} (logs: $LOGF; pid: $PIDF)"
nohup python -u /opt/airflow/dags/gradio_app.py >"$LOGF" 2>&1 & echo $! > "$PIDF"
sleep 2
echo "[gradio] PID: $(cat $PIDF)"
echo "[gradio] Si NO ves link público en el log, usa: http://localhost:${PORT}"
""",
        trigger_rule="none_failed_min_one_success",
    )
    # ================================================================

    start_pipeline >> create_folders_task >> download_data >> split_task >> train_task >> gradio_task >> end_pipeline