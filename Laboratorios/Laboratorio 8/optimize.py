# Importamos las librerías necesarias
import os
import json
import pickle
from datetime import datetime

import optuna
from optuna.samplers import TPESampler
import plotly.io as pio

import xgboost as xgb
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Definimos una función para recuperar el mejor modelo
def get_best_model(experiment_name: str):
    exp_local = mlflow.get_experiment_by_name(experiment_name)
    if exp_local is None:
        raise RuntimeError(f"Experimento '{experiment_name}' no encontrado en MLflow")
    runs = mlflow.search_runs(experiment_ids=[exp_local.experiment_id])
    if runs.empty or "metrics.valid_f1" not in runs.columns:
        raise RuntimeError("No se encontraron runs con la métrica 'valid_f1' en el experimento")
    best_run_id = runs.sort_values("metrics.valid_f1", ascending=False)["run_id"].iloc[0]
    return mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")

def optimize_model(n_trials: int = 50, random_state: int = 42):
    """
    1) Optimiza hiperparámetros de XGBoost usando Optuna.
    2) Registra cada trial como un run dentro de un experimento nuevo (nombres interpretables).
       Cada run registra la métrica 'valid_f1' y guarda el modelo con mlflow.sklearn.log_model.
    3) Guarda los gráficos de Optuna en plots/ y los sube a MLflow como artefactos.
    4) Recupera el mejor modelo (get_best_model), lo serializa con pickle en models/model.pkl
       y guarda ese pickle en los artefactos MLflow en 'models'.
    5) Registra versiones de librerías y guarda configuración final y la importancia de variables
       en plots/.
    Devuelve (experiment_name, best_model_loaded).
    """

    # Cargamos los datos
    df = pd.read_csv("water_potability.csv")
    X = df.drop("Potability", axis=1)
    y = df["Potability"]

    # Separamos los datos en train, val y test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=random_state, stratify=y_train_val
    )

    # Creamos un nuevo experimento en MLflow
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"XGBoost_Optuna_{timestamp}"
    mlflow.set_experiment(experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)

    # carpetas locales para artefactos
    os.makedirs("plots", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Definimos la función objetivo para Optuna
    def objective(trial):
        # Definimos el espacio de búsqueda de hiperparámetros
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }

        # Definimos el nombre de la run
        run_name = f"XGBoost_lr{params['learning_rate']:.3f}_md{params['max_depth']}_n{params['n_estimators']}"
        
        # Iniciamos una nueva run en MLflow
        with mlflow.start_run(run_name=run_name):
            # Registramos los hiperparámetros
            mlflow.log_params(params)

            # Entrenamos el modelo
            clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", **params)
            clf.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            preds = clf.predict(X_val)
            f1 = float(f1_score(y_val, preds))

            # Registramos la métrica de validación con el nombre requerido
            mlflow.log_metric("valid_f1", f1)

            # Registramos modelo en esta run para poder recuperarlo más tarde con runs:/<id>/model
            mlflow.sklearn.log_model(clf, artifact_path="model")

            # Guardamos la información del trial en un archivo JSON en plots/ tal como se pide
            trial_info = {"trial_number": trial.number, "valid_f1": f1, "params": params}
            trial_fn = f"plots/trial_{trial.number:03d}_info.json"
            with open(trial_fn, "w") as fh:
                json.dump(trial_info, fh)
            mlflow.log_artifact(trial_fn, artifact_path="plots")

        return f1

    # Ejecutamos la optimización con Optuna
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    # Guardamos los gráficos de Optuna en plots/ y los subimos a MLflow
    fig_hist = optuna.visualization.plot_optimization_history(study)
    pio.write_html(fig_hist, "plots/optuna_history.html", auto_open=False)

    fig_param = optuna.visualization.plot_param_importances(study)
    if fig_param is not None:
        pio.write_html(fig_param, "plots/optuna_param_importances.html", auto_open=False)

    # Guardamos los mejores hiperparámetros en un archivo JSON en plots/
    best_params = study.best_params
    with open("plots/best_params.json", "w") as fh:
        json.dump(best_params, fh)

    # Recuperamos el mejor modelo usando la función definida
    best_model_loaded = get_best_model(experiment_name)

    # Serializamos el modelo con pickle en models/model.pkl
    model_pickle_path = "models/model.pkl"
    with open(model_pickle_path, "wb") as fh:
        pickle.dump(best_model_loaded, fh)

    # Creamos una nueva run para guardar artefactos y metadata final
    final_run_name = f"artifacts_and_metadata_{timestamp}"
    with mlflow.start_run(run_name=final_run_name):
        # Guardamos las versiones de las librerías usadas
        mlflow.log_param("xgboost_version", xgb.__version__)
        mlflow.log_param("optuna_version", optuna.__version__)
        mlflow.log_param("mlflow_version", mlflow.__version__)
        mlflow.log_param("sklearn_version", sklearn.__version__)
        mlflow.log_param("pandas_version", pd.__version__)

        # Subimos los gráficos y archivos JSON generados en plots/
        mlflow.log_artifact("plots/optuna_history.html", artifact_path="plots")
        if os.path.exists("plots/optuna_param_importances.html"):
            mlflow.log_artifact("plots/optuna_param_importances.html", artifact_path="plots")
        mlflow.log_artifact("plots/best_params.json", artifact_path="plots")
        # subir todos los trial jsons generados
        for fn in sorted(os.listdir("plots")):
            if fn.startswith("trial_") and fn.endswith("_info.json"):
                mlflow.log_artifact(os.path.join("plots", fn), artifact_path="plots")

        # Importamos la importancia de variables y la guardamos en plots/
        try:
            if hasattr(best_model_loaded, "get_booster"):
                booster = best_model_loaded.get_booster()
                ax = xgb.plot_importance(booster, max_num_features=20)
            else:
                ax = xgb.plot_importance(best_model_loaded, max_num_features=20)
            plt.tight_layout()
            imp_path = "plots/importance_plot.png"
            plt.savefig(imp_path)
            plt.close()
            mlflow.log_artifact(imp_path, artifact_path="plots")
        except Exception:
            pass

        # Subimos el pickle del modelo
        mlflow.log_artifact(model_pickle_path, artifact_path="models")

    # Retornamos el nombre del experimento y el mejor modelo cargado
    print(f"Experiment saved: {experiment_name} (id={exp.experiment_id})")
    return experiment_name, best_model_loaded

if __name__ == "__main__":
    optimize_model()