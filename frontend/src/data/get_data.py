"""
Программа: Получение данных по пути и чтение
Версия: 1.0
"""

from io import BytesIO
import io
from typing import Dict, Tuple, Text
import streamlit as st
import pandas as pd


def get_dataset(dataset_path: Text) -> pd.DataFrame:
    """
    Получение данных по заданному пути
    :param dataset_path: путь до данных
    :return: датасет
    """
    return pd.read_parquet(dataset_path)


def data_merge(data_1: pd.DataFrame, data_2: pd.DataFrame) -> pd.DataFrame:
    """
    Объединяет два датасета по 'user_id' и делает начальную обработку
    """
    data = pd.merge(data_1, data_2, on="user_id")
    data = data[data.cpe_model_os_type != "Apple iOS"]
    data = data[data.is_male != "NA"]
    return data


def load_data(
    data: str, type_data: str
) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, BytesIO, str]]]:
    """
    Получение данных и преобразование в тип BytesIO для обработки в streamlit
    :param data: данные
    :param type_data: тип датасет (train/test)
    :return: датасет, датасет в формате BytesIO
    """
    dataset = pd.read_parquet(data)
    st.write("Dataset load")
    st.write(dataset.head())

    # Преобразовать dataframe в объект BytesIO (для последующего анализа в виде файла в FastAPI)
    dataset_bytes_obj = io.BytesIO()
    # запись в BytesIO буфер
    dataset.to_parquet(dataset_bytes_obj, index=False)
    # Сбросить указатель, чтобы избежать ошибки с пустыми данными
    dataset_bytes_obj.seek(0)

    files = {
        "file": (f"{type_data}_dataset.parquet", dataset_bytes_obj, "multipart/form-data")
    }
    return dataset, files
