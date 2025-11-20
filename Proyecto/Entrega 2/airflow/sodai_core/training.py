# airflow/sodai_core/training.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import logging

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score

import lightgbm as lgb
import optuna

from . import data_io
from . import features as feats


# ======================
# Construcción de preprocesador
# ======================

def build_preprocess_pipeline(
    num_strategy: str = "median",
    scaler: str = "standard",
    ohe_min_freq: float = 0.01,
    ohe_handle_unknown: str = "infrequent_if_exist",
) -> ColumnTransformer:
    """
    Construye el ColumnTransformer que se usó en la entrega 1
    (o uno equivalente), separado como función reutilizable.

    Ajusta los parámetros por defecto si en tu Optuna definiste
    otros por defecto.
    """
    num_imputer = SimpleImputer(strategy=num_strategy)

    if scaler == "standard":
        num_scaler = StandardScaler()
    elif scaler == "robust":
        num_scaler = RobustScaler()
    else:
        raise ValueError(f"Scaler no soportado: {scaler}")

    num_pipeline = Pipeline(
        steps=[
            ("imputer", num_imputer),
            ("scaler", num_scaler),
        ]
    )

    cat_encoder = OneHotEncoder(
        handle_unknown=ohe_handle_unknown,
        min_frequency=ohe_min_freq,
        sparse_output=True,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, feats.NUM_FEATURES),
            ("cat", cat_encoder, feats.CAT_FEATURES),
        ]
    )

    return preprocessor


# ======================
# Preparación de datos de entrenamiento a partir de tablas crudas
# ======================

def prepare_training_data_from_raw(
    data_dir: Path | str | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Realiza toda la preparación de datos empezando desde los datos crudos:

    - Carga transacciones, clientes y productos.
    - Limpia transacciones → transacciones_limpio.
    - Construye interacciones semanales → interacciones_semana.
    - Añade features de comportamiento.
    - Realiza holdout temporal train/valid/test.
    - Construye matrices X/y.

    Devuelve un diccionario con X_train, y_train, X_valid, y_valid, X_test, y_test, etc.
    """
    if data_dir is None:
        data_dir = data_io.DATA_DIR

    transacciones, clientes, productos = data_io.load_raw_tables(data_dir)

    transacciones_limpio = feats.clean_transactions(transacciones)
    interacciones = feats.build_weekly_interactions(
        transacciones_limpio, clientes, productos
    )
    interacciones = feats.add_behavioral_features(interacciones)

    train_df, valid_df, test_df = feats.temporal_holdout(interacciones)
    matrices = feats.build_ml_matrices(train_df, valid_df, test_df)

    return matrices


# ======================
# Entrenamiento Optuna + LGBM
# ======================

def _evaluate_predictions(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    # Manejar caso en que y_true tenga una sola clase (evita UndefinedMetricWarning)
    unique = np.unique(y_true)
    if unique.size < 2:
        # No se puede calcular AUC/PR cuando sólo hay una clase en y_true
        return {"auc": None, "pr_auc": None}

    auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    return {"auc": float(auc), "pr_auc": float(pr_auc)}


def optuna_lgbm_training(
    matrices: Dict[str, pd.DataFrame],
    n_trials: int = 30,
    random_state: int = 42,
    study_direction: str = "maximize",
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """
    Realiza la búsqueda de hiperparámetros de LGBM con Optuna
    usando PR-AUC en VALID como métrica objetivo.

    Devuelve un diccionario con:
      - best_preprocess: ColumnTransformer ajustado
      - best_model: LGBMClassifier ajustado
      - best_params: dict con hiperparámetros óptimos
      - metrics: dict con métricas en VALID y TEST
    """
    X_train = matrices["X_train"]
    y_train = matrices["y_train"].to_numpy()
    X_valid = matrices["X_valid"]
    y_valid = matrices["y_valid"].to_numpy()
    X_test = matrices["X_test"]
    y_test = matrices["y_test"].to_numpy()

    # Si no hay al menos dos clases en train no tiene sentido entrenar, a menos que se fuerce
    if np.unique(y_train).size < 2:
        if not force_retrain:
            logging.warning(
                "Skipping training: y_train contiene una sola clase. Use force_retrain=True para forzar entrenamiento."
            )
            metrics_valid = _evaluate_predictions(y_valid, np.zeros_like(y_valid, dtype=float))
            metrics_test = _evaluate_predictions(y_test, np.zeros_like(y_test, dtype=float))
            return dict(
                best_preprocess=None,
                best_model=None,
                best_params={},
                metrics=dict(valid=metrics_valid, test=metrics_test),
            )
        else:
            logging.warning(
                "Force training enabled but y_train contiene una sola clase. El modelo podrá no aprender la clase positiva."
            )

    EARLY_STOPPING_ROUNDS = 50

    def objective(trial: optuna.Trial) -> float:
        # Hiperparámetros de preprocesamiento
        num_strategy = trial.suggest_categorical("num_imputer", ["mean", "median"])
        scaler = trial.suggest_categorical("num_scaler", ["standard", "robust"])
        ohe_min_freq = trial.suggest_float("ohe_min_freq", 0.001, 0.02)
        ohe_drop_rare = trial.suggest_categorical("ohe_drop_rare", [True, False])

        handle_unknown = "infrequent_if_exist" if ohe_drop_rare else "ignore"

        # Hiperparámetros de LGBM
        params_lgb = dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 1000),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 31, 255),
            max_depth=trial.suggest_int("max_depth", 4, 16),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            random_state=random_state,
            n_jobs=-1,
            objective="binary",
        )

        # Build preprocessor and model. Fit preprocessor first and transform
        # train/valid so we can pass transformed eval_set to LGBM for early stopping.
        preprocessor = build_preprocess_pipeline(
            num_strategy=num_strategy,
            scaler=scaler,
            ohe_min_freq=ohe_min_freq,
            ohe_handle_unknown=handle_unknown,
        )

        # Fit preprocessor on training data and transform
        try:
            X_train_t = preprocessor.fit_transform(X_train)
            X_valid_t = preprocessor.transform(X_valid)
        except Exception:
            # If preprocessing fails for this trial, return a very low score
            return 0.0

        model = lgb.LGBMClassifier(**params_lgb, verbose=-1)

        try:
            model.fit(
                X_train_t,
                y_train,
                eval_set=[(X_valid_t, y_valid)],
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose=False,
            )
        except Exception:
            return 0.0

        proba_valid = model.predict_proba(X_valid_t)[:, 1]
        metrics_valid = _evaluate_predictions(y_valid, proba_valid)

        # Optuna maximiza esta métrica
        return metrics_valid["pr_auc"]

    study = optuna.create_study(direction=study_direction)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params

    # Reconstruimos el mejor modelo con los hiperparámetros óptimos
    num_strategy = best_params["num_imputer"]
    scaler = best_params["num_scaler"]
    ohe_min_freq = best_params["ohe_min_freq"]
    ohe_drop_rare = best_params["ohe_drop_rare"]
    handle_unknown = "infrequent_if_exist" if ohe_drop_rare else "ignore"

    preprocessor = build_preprocess_pipeline(
        num_strategy=num_strategy,
        scaler=scaler,
        ohe_min_freq=ohe_min_freq,
        ohe_handle_unknown=handle_unknown,
    )

    params_lgb = dict(
        n_estimators=best_params["n_estimators"],
        learning_rate=best_params["learning_rate"],
        num_leaves=best_params["num_leaves"],
        max_depth=best_params["max_depth"],
        min_child_samples=best_params["min_child_samples"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
        reg_alpha=best_params["reg_alpha"],
        reg_lambda=best_params["reg_lambda"],
        random_state=random_state,
        n_jobs=-1,
        objective="binary",
    )

    # Fit preprocessor and transform datasets, then train final model with early stopping
    preprocessor = build_preprocess_pipeline(
        num_strategy=num_strategy,
        scaler=scaler,
        ohe_min_freq=ohe_min_freq,
        ohe_handle_unknown=handle_unknown,
    )

    X_train_t = preprocessor.fit_transform(X_train)
    X_valid_t = preprocessor.transform(X_valid)
    X_test_t = preprocessor.transform(X_test)

    model = lgb.LGBMClassifier(**params_lgb, verbose=-1)

    try:
        model.fit(
            X_train_t,
            y_train,
            eval_set=[(X_valid_t, y_valid)],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=False,
        )
    except Exception:
        # If final fit fails, return metrics as None
        metrics_valid = _evaluate_predictions(y_valid, np.zeros_like(y_valid, dtype=float))
        metrics_test = _evaluate_predictions(y_test, np.zeros_like(y_test, dtype=float))
        return dict(
            best_preprocess=preprocessor,
            best_model=model,
            best_params=best_params,
            metrics=dict(valid=metrics_valid, test=metrics_test),
        )

    proba_valid = model.predict_proba(X_valid_t)[:, 1]
    proba_test = model.predict_proba(X_test_t)[:, 1]

    metrics_valid = _evaluate_predictions(y_valid, proba_valid)
    metrics_test = _evaluate_predictions(y_test, proba_test)

    return dict(
        best_preprocess=preprocessor,
        best_model=model,
        best_params=best_params,
        metrics=dict(valid=metrics_valid, test=metrics_test),
    )


# ======================
# Función de alto nivel para el DAG
# ======================

def run_optuna_training_and_save(
    data_dir: Path | str | None = None,
    out_dir: Path | str | None = None,
    n_trials: int = 30,
    prefix: str = "best",
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """
    Función de alto nivel para:

    - Preparar datos desde parquet.
    - Ejecutar Optuna + LGBM.
    - Guardar artefactos en disco.
    - Devolver resumen de resultados (incluyendo paths a los artefactos).

    Esta es la función ideal para llamar desde una task de Airflow.
    """
    if data_dir is None:
        data_dir = data_io.DATA_DIR
    if out_dir is None:
        out_dir = data_io.ARTIFACTS_OPTUNA_DIR

    data_dir = Path(data_dir)
    out_dir = Path(out_dir)

    # 1) Preparar datos y entrenar con Optuna
    matrices = prepare_training_data_from_raw(data_dir=data_dir)
    result = optuna_lgbm_training(matrices, n_trials=n_trials, force_retrain=force_retrain)

    metadata = dict(
        best_params=result.get("best_params", {}),
        metrics=result.get("metrics", {}),
        # Se pueden añadir más campos: fecha de entrenamiento, versión de datos, etc.
    )

    # Si el entrenamiento fue saltado (por ejemplo, una sola clase en y),
    # no intentamos guardar artefactos y devolvemos una salida indicando
    # que el reentrenamiento fue omitido.
    if result.get("best_model") is None:
        logging.warning("Training skipped: no model was trained (single-class target).")
        out_dict: Dict[str, Any] = dict(
            best_params=result.get("best_params", {}),
            metrics=result.get("metrics", {}),
            artifacts_dir=str(out_dir),
            model_path=None,
            preprocess_path=None,
            metadata_path=None,
            skipped_due_to_single_class=True,
        )
        return out_dict

    # 2) Guardar artefactos en disco
    data_io.save_model_artifacts(
        preprocess=result["best_preprocess"],
        model=result["best_model"],
        metadata=metadata,
        out_dir=out_dir,
        prefix=prefix,
    )

    # 3) Intentar descubrir automáticamente los paths de modelo y preprocesador
    model_path: Optional[Path] = None
    preprocess_path: Optional[Path] = None
    metadata_path: Optional[Path] = None

    # Primero intentamos las rutas explícitas que usamos al guardar
    candidate_preprocess = out_dir / f"preprocess_{prefix}.joblib"
    candidate_model = out_dir / f"model_lgbm_{prefix}.joblib"
    candidate_metadata = out_dir / f"metadata_{prefix}.json"

    if candidate_model.exists():
        model_path = candidate_model
    if candidate_preprocess.exists():
        preprocess_path = candidate_preprocess
    if candidate_metadata.exists():
        metadata_path = candidate_metadata

    # Fallback: buscar por glob si algún archivo no fue encontrado
    if out_dir.exists():
        if model_path is None:
            for p in out_dir.glob(f"*{prefix}*model*.joblib"):
                model_path = p
                break
        if preprocess_path is None:
            for p in out_dir.glob(f"*{prefix}*preprocess*.joblib"):
                preprocess_path = p
                break
        if metadata_path is None:
            # buscamos cualquier json que contenga el prefijo o el nombre estándar
            for p in out_dir.glob(f"*{prefix}*.json"):
                metadata_path = p
                break
            if metadata_path is None:
                std_meta = out_dir / "metadata.json"
                if std_meta.exists():
                    metadata_path = std_meta

    # 4) Construimos el diccionario de salida que espera el DAG
    out_dict: Dict[str, Any] = dict(
        # best_preprocess=result["best_preprocess"],
        # best_model=result["best_model"],
        best_params=result["best_params"],
        metrics=result["metrics"],
        artifacts_dir=str(out_dir),
        model_path=str(model_path) if model_path is not None else None,
        preprocess_path=str(preprocess_path) if preprocess_path is not None else None,
        metadata_path=str(metadata_path) if metadata_path is not None else None,
    )

    return out_dict


def retrain_model_with_optuna(
    *args,
    **kwargs,
) -> Dict[str, Any]:
    """
    Wrapper de compatibilidad para el DAG `sodai_ml_pipeline`.

    - Acepta los mismos argumentos que `run_optuna_training_and_save`.
    - Tolera los nombres:
        * artifacts_dir
        * artifacts_entrega1_dir
        * artifacts_output_dir
      como alias de `out_dir`.
    - Reutiliza toda la lógica existente de entrenamiento con Optuna.
    """
    # 1) Mapear aliases de directorios de artefactos a out_dir
    alias_keys = ("artifacts_output_dir", "artifacts_entrega1_dir", "artifacts_dir")

    # Si no hay out_dir explícito, usar el primero de los alias que encontremos
    if "out_dir" not in kwargs:
        for alias in alias_keys:
            if alias in kwargs:
                kwargs["out_dir"] = kwargs[alias]
                break

    # En cualquier caso, eliminamos los aliases para que no lleguen a run_optuna_training_and_save
    for alias in alias_keys:
        kwargs.pop(alias, None)

    # 2) Filtrar kwargs para que solo pasen los parámetros que realmente existen
    #    en run_optuna_training_and_save
    import inspect

    sig = inspect.signature(run_optuna_training_and_save)
    allowed_params = set(sig.parameters.keys())
    clean_kwargs = {k: v for k, v in kwargs.items() if k in allowed_params}

    # 3) Delegar al pipeline completo ya implementado
    return run_optuna_training_and_save(*args, **clean_kwargs)
