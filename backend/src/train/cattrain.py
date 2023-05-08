"""
Программа: Тренировка данных
Версия: 1.0
"""

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    log_loss,
    f1_score,
)

from catboost import CatBoostClassifier

from typing import Dict

from ..train.metrics import save_metrics

import warnings

warnings.filterwarnings("ignore")

RAND = 10
n_folds = 5


def find_optimal_params(
        x_train: pd.DataFrame, y_train: pd.Series
) -> dict:
    """
    Пайплайн для получения лучших параметров
    :param data_train: датасет train
    :param data_test: датасет test
    :return: словарь с лучшими параметрами
    """
    grid = {
        "n_estimators": [1000],
        "learning_rate": np.linspace(0.01, 0.1, 5),
        "boosting_type": ["Ordered", "Plain"],
        "max_depth": list(range(3, 12)),
        "l2_leaf_reg": np.logspace(-5, 2, 5),
        "random_strength": list(range(10, 50, 5)),
        "bootstrap_type": ["Bayesian", "Bernoulli", "MVS", "No"],
        "border_count": [128, 254],
        "grow_policy": ["SymmetricTree", "Depthwise", "Lossguide"],
        "random_state": [RAND],
    }

    model = CatBoostClassifier(silent=True)
    grid_search_result = model.randomized_search(
        grid, X=x_train, y=y_train, verbose=False
    )
    cat_best = grid_search_result['params']
    return cat_best


def train_val_cat(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    clf,
    metric_path: str,
    params: dict,
    cat_features: list = None,
    eval_metric: str = None,
    early_stop: bool = False,
    early_stopping_rounds: int = 100,
    num_folds: int = n_folds,
    random_state: int = RAND,
    shuffle: bool = True,
):
    """
    Получает результаты при помощи кросс-валидации для задачи классиификации
    :param x_train: датасет
    :param x_test: датасет
    :param y_train: датасет
    :param y_test: датасет
    :param clf: модель обучения
    :param params: словарь с лучшими параметрами
    :param cat_features: список индексов категориальных признаков
    :param eval_metric: метрика, используемая для обнаружения переобучения
    :param early_stop: параметр ранней остановы
    :param early_stopping_rounds: параметр ранней остановы
    :param num_folds: количество фолдов
    :param random_state: random_state
    :param shuffle: параметр перемешивания данных
    :return: CatBoostClassifier
    """
    folds = StratifiedKFold(
        n_splits=num_folds, random_state=random_state, shuffle=shuffle
    )
    pred_test = []
    pred_prob_test = []

    for fold, (train_index, test_index) in enumerate(folds.split(x_train, y_train)):
        X_train_, X_val = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_, y_val = y_train.iloc[train_index], y_train.iloc[test_index]

        model = clf(**params)

        if early_stop == True:
            if eval_metric is None:
                model.fit(
                    X_train_,
                    y_train_,
                    cat_features=cat_features,
                    silent=True,
                    early_stopping_rounds=early_stopping_rounds,
                )
            else:
                model.fit(
                    X_train_,
                    y_train_,
                    eval_metric=eval_metric,
                    silent=True,
                    cat_features=cat_features,
                    early_stopping_rounds=early_stopping_rounds,
                )
        else:
            model.fit(X_train_, y_train_, cat_features=cat_features)

        y_pred = model.predict(x_test)
        y_pred_prob = model.predict_proba(x_test)

        # holdout list
        pred_test.append(y_pred)
        pred_prob_test.append(y_pred_prob)

    fin_test_pred = stats.mode(np.column_stack(pred_test), axis=1)[0]
    fin_test_pred_prob = np.mean(pred_prob_test, axis=0)

    # save metrics
    save_metrics(y_test=y_test,
                 y_pred=fin_test_pred,
                 y_proba=fin_test_pred_prob,
                 metric_path=metric_path)
    return model
