"""
Программа: Получение предсказания на основе обученной модели
Версия: 1.0
"""

import os
import yaml
import joblib
import pandas as pd
from ..data.get_data import get_dataset
from ..transform.transform import pipeline_preprocess


def pipeline_evaluate(
    config_path, dataset: pd.DataFrame = None, data_path: str = None, **kwargs
) -> list:
    """
    Предобработка входных данных и получение предсказаний
    :param dataset: датасет
    :param config_path: путь до конфигурационного файла
    :param data_path: путь до файла с данными
    :return: предсказания
    """
    # get params
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    preprocessing_config = config["preprocessing"]
    train_config = config["train"]

    # preprocessing
    if data_path:
        dataset = get_dataset(dataset_path=data_path)

    dataset = pipeline_preprocess(data=dataset, **preprocessing_config)
    dataset = dataset.drop(columns=train_config["target_column"], axis=1)

    model = joblib.load(os.path.join(train_config["model_path"]))
    prediction = model.predict(dataset).tolist()

    return prediction
