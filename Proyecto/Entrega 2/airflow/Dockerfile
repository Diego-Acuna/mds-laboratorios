# Imagen base oficial de Airflow con Python 3.11
FROM apache/airflow:2.9.3-python3.11

# Cambiamos a root para poder instalar paquetes de sistema si fuese necesario
USER root

# Paquetes del sistema útiles para compilar algunas librerías científicas
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Volvemos al usuario airflow, que es el que usa la imagen base
USER airflow

# Instalamos las dependencias de ML y MLOps que usa el proyecto
RUN pip install --no-cache-dir \
      numpy \
      pandas \
      scikit-learn \
      lightgbm \
      xgboost \
      shap \
      optuna \
      mlflow \
      pyarrow \
      fastparquet \
      joblib

# Copiamos la librería interna sodai_core al path de Airflow para poder importarla
# Nota: esto copia el código tal como está en el contexto de build (carpeta airflow/)
COPY sodai_core /opt/airflow/sodai_core

# Añadimos /opt/airflow al PYTHONPATH para asegurarnos de que sodai_core es importable
ENV PYTHONPATH=/opt/airflow

# Directorio de trabajo por defecto
WORKDIR /opt/airflow