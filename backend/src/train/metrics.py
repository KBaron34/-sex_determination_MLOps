"""
Программа: Получение метрик
Версия: 1.0
"""

import json
import yaml
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)


def get_metrics(y_test: pd.Series, y_pred: pd.Series, y_proba: pd.Series) -> dict:
    '''
    Создает словарь с оновными метриками
    :param y_test: реальные данные
    :param y_pred: предсказанные значения
    :param y_proba: предсказанные вероятности
    :return: словарь с метриками
    '''
    dict_metrics = {
        'accuracy': round(accuracy_score(y_test, y_pred), 3),
        'roc_auc': round(roc_auc_score(y_test, y_proba[:, 1]), 3),
        'precision': round(precision_score(y_test, y_pred), 3),
        'recall': round(recall_score(y_test, y_pred), 3),
        'f1': round(f1_score(y_test, y_pred), 3),
        'logloss': round(log_loss(y_test, y_proba), 3)
    }
    return dict_metrics


def save_metrics(
    y_test: pd.Series, y_pred: pd.Series, y_proba: pd.Series, metric_path: str
) -> None:
    """
    Получает и сохраненяет метрики
    :param y_test: реальные данные
    :param y_pred: предсказанные значения
    :param y_proba: предсказанные вероятности
    :param metric_path: путь для сохранения метрик
    """
    result_metrics = get_metrics(
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_proba
    )
    with open(metric_path, "w") as file:
        json.dump(result_metrics, file)


def load_metrics(config_path: str) -> dict:
    """
    Получает метрики из файла
    :param config_path: путь до конфигурационного файла
    :return: метрики
    """
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    with open(config["train"]["metrics_path"]) as json_file:
        metrics = json.load(json_file)

    return metrics
