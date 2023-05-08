"""
Программа: Отрисовка слайдеров и кнопок для ввода данных
с дальнейшим получением предсказания на основании введенных значений
Версия: 1.0
"""

import json
from io import BytesIO
import pandas as pd
import requests
import streamlit as st


def evaluate_input(unique_data_path: str, endpoint: object) -> None:
    """
    Получает входные данные путем ввода в UI -> выводит результат
    :param unique_data_path: путь до уникальных значений
    :param endpoint: endpoint
    """
    with open(unique_data_path) as file:
        unique_df = json.load(file)

    # поля для вводы данных, используем уникальные значения
    region_name = st.sidebar.selectbox("region_name", (unique_df["region_name"]))
    city_name = st.sidebar.selectbox("city_name", (unique_df["city_name"]))
    cpe_manufacturer_name = st.sidebar.selectbox(
        "cpe_manufacturer_name", (unique_df["cpe_manufacturer_name"])
    )
    cpe_model_name = st.sidebar.selectbox(
        "cpe_model_name", (sorted(unique_df["cpe_model_name"]))
    )
    url_host = st.sidebar.selectbox("url_host", (unique_df["url_host"]))
    cpe_type_cd = st.sidebar.selectbox("cpe_type_cd", (unique_df["cpe_type_cd"]))
    cpe_model_os_type = st.sidebar.selectbox(
        "cpe_model_os_type", (unique_df["cpe_model_os_type"])
    )
    price = st.sidebar.number_input(
        "price",
        min_value=min(unique_df["price"]),
        max_value=max(unique_df["price"]),
    )
    date = st.sidebar.text_input("date")
    part_of_day = st.sidebar.selectbox(
        "part_of_day", (unique_df["part_of_day"])
    )
    request_cnt = st.sidebar.number_input(
        "request_cnt",
        min_value=min(unique_df["request_cnt"]),
        max_value=max(unique_df["request_cnt"]),
    )
    user_id = st.sidebar.selectbox(
        "user_id", (unique_df["user_id"])
    )

    dict_data = {
        "region_name": region_name,
        "city_name": city_name,
        "cpe_manufacturer_name": cpe_manufacturer_name,
        "cpe_model_name": cpe_model_name,
        "url_host": url_host,
        "cpe_type_cd": cpe_type_cd,
        "cpe_model_os_type": cpe_model_os_type,
        "price": price,
        "date": date,
        "part_of_day": part_of_day,
        "request_cnt": request_cnt,
        "user_id": user_id
    }

    st.write(
        f"""### Данные клиента:\n
    1) region_name: {dict_data['region_name']}
    2) city_name: {dict_data['city_name']}
    3) cpe_manufacturer_name: {dict_data['cpe_manufacturer_name']}
    4) cpe_model_name: {dict_data['cpe_model_name']}
    5) url_host: {dict_data['url_host']}
    6) cpe_type_cd: {dict_data['cpe_type_cd']}
    7) cpe_model_os_type: {dict_data['cpe_model_os_type']}
    8) price: {dict_data['price']}
    9) date: {dict_data['date']}
    10) part_of_day: {dict_data['part_of_day']}
    11) request_cnt: {dict_data['request_cnt']}
    12) user_id: {dict_data['user_id']}
    """
    )

    # evaluate and return prediction (text)
    button_ok = st.button("Predict")
    if button_ok:
        result = requests.post(endpoint, timeout=8000, json=dict_data)
        json_str = json.dumps(result.json())
        output = json.loads(json_str)
        st.write(f"## {output[0]}")
        st.success("Success!")


def evaluate_from_file(data: pd.DataFrame, endpoint: object, files: BytesIO):
    """
    Получение входных данных в качестве файла -> вывод результата в виде таблицы
    :param data: датасет
    :param endpoint: endpoint
    :param files:
    """
    button_ok = st.button("Predict")
    if button_ok:
        data_ = data[:5]
        output = requests.post(endpoint, files=files, timeout=8000)
        data_["predict"] = output.json()["prediction"]
        st.write(data_.head())
