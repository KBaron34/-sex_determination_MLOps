"""
Программа: Frontend часть проекта
Версия: 1.0
"""


import os
import yaml
import streamlit as st
import pandas as pd
from src.data.get_data import load_data, get_dataset, data_merge
from src.plotting.charts import barplot_group, boxplot_group
from src.train.training import start_training
from src.evaluate.evaluate import evaluate_from_file, evaluate_input

CONFIG_PATH = '../config/params.yml'


def main_page():
    """
    Страница с описанием проекта
    """
    st.markdown('# Описание проекта')
    st.write(
        """Ссылка на данные: https://ods.ai/competitions/mtsmlcup
    
            Задача состоит в определении пола пользователя 
    по его цифровым следам. В Digital-рекламе часто сегмент включает с ебя пол. 
    Эта задача особенно актуальна для рекламных DSP-площадок,
    которые в OpenRTB запросах получают такие данные со всех сайтов,
    размещающих рекламу за деньги.""")

    st.markdown(
        """
        ### Описание данных:
            - region_name – Регион;
            - city_name – Населенный пункт;
            - cpe_manufacturer_name – Производитель устройства;
            - cpe_model_name – Модель устройства;
            - url_host – Домен, с которого пришел рекламный запрос;
            - cpe_type_cd – Тип устройства (смартфон или что-то другое);
            - cpe_model_os_type – Операционка на устройстве;
            - price – Цена цены устройства;
            - date – Дата;
            - part_of_day – Время дня (утро, вечер, итд);
            - request_cn – Число запросов одного пользователя за время дня (поле part_of_day);
            - user_id – ID пользователя;
            - age – Возраст пользователя;
            - is_male – Пол пользователя : мужчина (1-Да, 0-Нет).""")

    st.markdown('### Target - is_male')


def exploratory():
    """
    Exploratory data analysis
    """
    st.markdown("# Exploratory data analysis")

    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    data = get_dataset(dataset_path=config["preprocessing"]["train_data_path"])
    data = data_merge(data_1=data,
                      data_2=pd.read_parquet(config["preprocessing"]["target_train_path"]))
    st.write(data.head())

    # plotting with checkbox
    male_manufacturer = st.sidebar.checkbox('Зависимость пола от производителя устройства')
    male_request = st.sidebar.checkbox('Зависимость пола от количества запросов в день')
    male_part_day = st.sidebar.checkbox('Зависимость пола от активности в течение дня')
    male_month = st.sidebar.checkbox('Зависимость пола от месяца')

    if male_manufacturer:
        st.pyplot(barplot_group(col_main='cpe_manufacturer_name',
                                col_group='is_male',
                                data=data,
                                title='Зависимость пола от производитель устройства',
                                grad=90))
        st.pyplot(barplot_group(col_main='is_male',
                                col_group='cpe_manufacturer_name',
                                data=data,
                                title='Зависимость пола от производитель устройства',
                                grad=90))
        st.write("""Производитель устройства и его модель 
                    частично влияют на принадлежность к тому или иному полу.""")
    if male_request:
        st.pyplot(boxplot_group(targ_main='is_male',
                                data=data,
                                exp_group='request_cnt',
                                title='Зависимость пола от количества запросов в день'))
        st.write("""В среднем, вероятность сделать определенное колличество запросов в день - одинакова.""")
    if male_part_day:
         st.pyplot(barplot_group(col_main='part_of_day',
                                 col_group='is_male',
                                 data=data,
                                 title='Зависимость пола от активности в течение дня',
                                 grad=90))
         st.write("""Пик активности женщин в дневное и вечернее время, 
                     а мужчины - утреннее и ночное.""")
    if male_month:
        month_dict = {1: 'January',
                      2: 'February',
                      3: 'March',
                      4: 'April',
                      5: 'May',
                      6: 'June',
                      7: 'July',
                      8: 'August',
                      9: 'September',
                      10: 'October',
                      11: 'November',
                      12: 'December'}
        data['month'] = pd.to_datetime(data.date).dt.month.map(month_dict)
        st.pyplot(barplot_group(col_main='month',
                                col_group='is_male',
                                data=data,
                                title='Зависимость пола от месяца',
                                grad=0))
        st.write("""
                От месяца зависит принадлежность к полу. 
                Так в декабре, ноябре и сентябре пик активности у мужчин. 
                А в июле, августе и июне у женщин.""")


def training():
    """
    Тренировка модели
    """
    st.markdown("# Training model")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config['endpoints']['train']

    if st.button('Start training'):
        start_training(config=config, endpoint=endpoint)


def prediction():
    """
    Получает предсказания путем ввода данных
    """
    st.markdown("# Prediction")
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["prediction_input"]
    unique_data_path = config["preprocessing"]["unique_values_path"]

    # проверка на наличие сохраненной модели
    if os.path.exists(config["train"]["model_path"]):
        evaluate_input(unique_data_path=unique_data_path, endpoint=endpoint)
    else:
        st.error("Сначала обучите модель")


def prediction_from_file():
    """
    Получение предсказаний из файла с данными
    """
    st.markdown('# Prediction')
    with open(CONFIG_PATH) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config['endpoints']['prediction_from_file']

    upload_file = st.file_uploader("", type=["xlsx", "parquet"],
                                   accept_multiple_files=False)
    if upload_file:
        dataset_df, files = load_data(data=upload_file, type_data="Test")
        if os.path.exists(config["train"]["model_path"]):
            evaluate_from_file(data=dataset_df, endpoint=endpoint, files=files)
        else:
            st.error("Сначала обучите модель")


def main():
    """
    Сборка пайплайна в одном блоке
    """
    page_names_to_funcs = {
        'Описание проекта': main_page,
        'Exploratory data analysis': exploratory,
        'Training model': training,
        'Prediction': prediction,
        'Prediction from file': prediction_from_file
    }
    select_page = st.sidebar.selectbox('Выберите пукт', page_names_to_funcs.keys())
    page_names_to_funcs[select_page]()


if __name__ == "__main__":
    main()
