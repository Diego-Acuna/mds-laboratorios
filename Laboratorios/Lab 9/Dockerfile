FROM apache/airflow:2.9.2

USER root
RUN apt-get update && apt-get install -y --no-install-recommends curl \
  && rm -rf /var/lib/apt/lists/*

USER airflow
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scikit-learn==1.5.1 \
    joblib==1.4.2 \
    "huggingface_hub==0.24.6" \
    requests==2.31.0 \
    packaging==24.0 \
    fsspec==2023.12.2 \
    gradio==4.44.0