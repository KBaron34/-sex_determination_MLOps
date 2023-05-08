"""
Программа: Тренировка данных
Версия: 1.0
"""

import optuna
from lightgbm import LGBMClassifier
from optuna import Study

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from ..data.split_dataset import get_train_test_data
from ..train.metrics import save_metrics


def objective(
    trial,
    data_x: pd.DataFrame,
    data_y: pd.Series,
    n_folds: int = 5,
    random_state: int = 10
) -> np.array:
    """
    Целевая функция для поиска параметров
    :param trial: кол-во trials
    :param data_x: данные объект-признаки
    :param data_y: данные с целевой переменной
    :param n_folds: кол-во фолдов
    :param random_state: random_state
    :return: среднее значение метрики по фолдам
    """
    param_grid = {
        "n_estimators": trial.suggest_categorical("n_estimators", [100]),
        "random_state": trial.suggest_categorical("random_state", [random_state]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 1000, step=20),
        "max_depth": trial.suggest_int("max_depth", 4, 15, step=1),
        "lambda_l1": trial.suggest_int("lambda_l1", 0.0, 100),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100),
        "min_gain_to_split": trial.suggest_int("min_gain_to_split", 0, 20),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 1.0),
        "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1.0),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced"]),
    }

    cv_folds = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=random_state
    )
    cv_predicts = np.empty(n_folds)

    for idx, (train_idx, test_idx) in enumerate(cv_folds.split(data_x, data_y)):
        x_train, x_test = data_x.iloc[train_idx], data_x.iloc[test_idx]
        y_train, y_test = data_y.iloc[train_idx], data_y.iloc[test_idx]

        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")
        model = LGBMClassifier(**param_grid, silent=True)
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_test, y_test)],
            eval_metric="auc",
            early_stopping_rounds=100,
            callbacks=[pruning_callback],
            verbose=-1,
        )
        predict = model.predict_proba(x_test)[:, 1]
        cv_predicts[idx] = roc_auc_score(y_test, predict)
    return np.mean(cv_predicts)


def find_optimal_params(
    data_train: pd.DataFrame, data_test: pd.DataFrame, **kwargs
) -> Study:
    """
    Пайплайн для тренировки модели
    :param data_train: датасет train
    :param data_test: датасет test
    :return: [LGBMClassifier tuning, Study]
    """
    x_train, x_test, y_train, y_test = get_train_test_data(
        data_train=data_train, data_test=data_test, target=kwargs["target_column"]
    )

    study = optuna.create_study(direction="maximize", study_name="LGB")
    function = lambda trial: objective(
        trial, x_train, y_train, kwargs["n_folds"], kwargs["random_state"]
    )
    study.optimize(function, n_trials=kwargs["n_trials"], show_progress_bar=True)
    return study


def train_model(
    data_train: pd.DataFrame,
    data_test: pd.DataFrame,
    study: Study,
    target: str,
    metric_path: str,
) -> LGBMClassifier:
    """
    Обучение модели на лучших параметрах
    :param data_train: тренировочный датасет
    :param data_test: тестовый датасет
    :param study: study optuna
    :param target: название целевой переменной
    :param metric_path: путь до папки с метриками
    :return: LGBMClassifier
    """
    # get data
    x_train, x_test, y_train, y_test = get_train_test_data(
        data_train=data_train, data_test=data_test, target=target
    )

    # training optimal params
    clf = LGBMClassifier(**study.best_params, silent=True, verbose=-1)
    clf.fit(x_train,
            y_train,
            eval_set=[(x_test, y_test)],
            verbose=-1)

    # save metrics
    save_metrics(y_test=y_test,
                 y_pred=clf.predict(x_test),
                 y_proba=clf.predict_proba(x_test),
                 metric_path=metric_path)
    return clf
