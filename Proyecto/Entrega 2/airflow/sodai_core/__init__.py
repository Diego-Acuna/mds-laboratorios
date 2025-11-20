# airflow/sodai_core/__init__.py

"""
Paquete principal de SodAI Drinks para reutilizar lógica de ML
entre notebooks, scripts de entrenamiento e integración con Airflow.
"""

from . import data_io
from . import features
from . import training
from . import inference


__all__ = [
    "data_io",
    "features",
    "training",
    "inference",
]