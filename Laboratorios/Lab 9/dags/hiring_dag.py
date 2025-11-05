# dags/hiring_dag.py
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

from hiring_functions import create_folders

with DAG(
    dag_id="hiring_linear_pipeline",
    start_date=datetime(2025, 11, 1),
    schedule_interval="@daily",
    catchup=False,                      # corre sólo ejecuciones “nuevas”
    default_args={"owner": "airflow"},
    tags=["lab9", "mlops", "hiring"],
) as dag:

    create_folders_task = PythonOperator(
        task_id="create_folders",
        python_callable=create_folders,
        op_kwargs={
            "base_dir": "/opt/airflow",       # en Docker oficial
            "ds_nodash": "{{ ds_nodash }}",   # pasa la fecha de ejecución (Jinja)
        },
    )

    # más adelante encadenarás aquí: split_data >> preprocess_and_train >> gradio
    # por ahora, sólo registramos la tarea de creación de carpetas
    create_folders_task
