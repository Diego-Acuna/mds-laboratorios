"""
Quick training helper
Run inside the airflow-scheduler container to train a small model on current data
and overwrite `/opt/airflow/models/*` artifacts so inference can proceed.
Usage:
    docker compose exec airflow-scheduler bash -lc "python /opt/airflow/scripts/quick_train_and_save.py"
"""
from pathlib import Path
import logging

from sodai_core import training, data_io
import lightgbm as lgb

logging.basicConfig(level=logging.INFO)

OUT_DIR = Path('/opt/airflow/models')

if __name__ == '__main__':
    logging.info('Preparing training matrices from raw data')
    matrices = training.prepare_training_data_from_raw()
    X_train = matrices['X_train']
    y_train = matrices['y_train'].to_numpy()
    X_valid = matrices['X_valid']
    y_valid = matrices['y_valid'].to_numpy()

    logging.info('Building preprocess pipeline and transforming data')
    preproc = training.build_preprocess_pipeline()
    X_train_t = preproc.fit_transform(X_train)
    X_valid_t = preproc.transform(X_valid)

    logging.info('Training small LightGBM model')
    model = lgb.LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train_t, y_train, eval_set=[(X_valid_t, y_valid)], early_stopping_rounds=20, verbose=False)

    logging.info('Saving artifacts to %s', OUT_DIR)
    metadata = {'note': 'quick_fit overwrite', 'n_estimators': 200}
    data_io.save_model_artifacts(preprocess=preproc, model=model, metadata=metadata, out_dir=OUT_DIR, prefix='best')

    logging.info('Done')
