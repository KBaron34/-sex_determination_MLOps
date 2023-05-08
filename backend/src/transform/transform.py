"""
Программа: Предобработка данных
Версия: 1.0
"""

import json
import pandas as pd
import pyarrow as pa
import numpy as np
import scipy.sparse
import implicit
import warnings

warnings.filterwarnings("ignore")


def data_merge(data_1: pd.DataFrame, data_2: pd.DataFrame) -> pd.DataFrame:
    """
    Объединяет два датасета по 'user_id' и делает начальную обработку
    """
    data = pd.merge(data_1, data_2[["user_id", "age"]], on="user_id")
    data = data[data.cpe_model_os_type != "Apple iOS"]
    return data


def filling_pass(data: pd.DataFrame, list_non: list) -> pd.DataFrame:
    """
    Заполняет пропуски значений в датасете
    :param data: датасет
    :param list_non: список с признаками в которых есть пропуски
    :return: датасет
    """
    for i in list_non:
        if i == "price":
            data[i] = data[i].fillna(data[i].median())
        elif i == "age":
            data[i] = data[i].fillna(data[i].mode().iloc[0])
            data.loc[data[i] == 0, i] = data[i].mode().iloc[0]
    return data


def replace_values(data: pd.DataFrame, map_change_columns: dict) -> pd.DataFrame:
    """
    Заменяет значения в датасете
    :param data: датасет
    :param map_change_columns: словарь с признаками и значениями
    :return: датасет
    """
    data["month"] = pd.to_datetime(data.date).dt.month
    return data.replace(map_change_columns)


def get_bins(data: pd.DataFrame, first_val: int, second_val: int) -> pd.DataFrame:
    """
    Генерирует бины для разных признаков
    :param data: датасет
    :param first_val: первый порог значения для разбиения на бины
    :param second_val: второй порог значения для разбиения на бины
    :return: датасет
    """
    result = (
        "small"
        if data <= first_val
        else "medium"
        if first_val < data <= second_val
        else "large"
    )
    return result


def data_drop(data: pd.DataFrame, drop_columns: list) -> pd.DataFrame:
    """
    Удаляет признаки
    :param data: датасет
    :param drop_columns: список с признаками
    :return: датасет
    """
    return data.drop(columns=drop_columns, axis=1)


def transform_types(data: pd.DataFrame, change_type_columns: dict) -> pd.DataFrame:
    """
    Преобразазует признаки в заданный тип данных
    :param data: датасет
    :param change_type_columns: словарь с признаками и типами данных
    :return:
    """
    return data.astype(change_type_columns, errors="raise")


def check_columns_evaluate(data: pd.DataFrame, unique_values_path: str) -> pd.DataFrame:
    """
    Проверка на наличие признаков из train и упорядочивание признаков согласно train
    :param data: датасет test
    :param unique_values_path: путь до списока с признаками train для сравнения
    :return: датасет test
    """
    with open(unique_values_path) as json_file:
        unique_values = json.load(json_file)

    column_sequence = unique_values.keys()

    return data[column_sequence]


def data_aggregate(data: pd.DataFrame, list_select, list_group) -> pa.Table:
    """
    Преобразует в формат pa.Table, группирует и аггрегирует данные
    :param data: датасет
    :param list_select: используемые признаки из датасета
    :param list_group: группируемые признаки
    :return: pa.Table
    """
    fin_df_pq = pa.Table.from_pandas(data)
    data_agg = (
        fin_df_pq.select(list_select)
        .group_by(list_group)
        .aggregate(
            [
                ("month", "count_distinct"),
                ("city_name", "count_distinct"),
                ("cpe_model_name", "count_distinct"),
                ("bins_price", "count_distinct"),
                ("part_of_day", "count_distinct"),
                ("bins_age", "count_distinct"),
            ]
        )
    )
    return data_agg


def gen_mat(
    data_agg: pa.Table, usr_dict: dict, url_dict: dict, dat: str, user: str, url: str
) -> scipy.sparse.coo_matrix:
    """
    Генерирует разряженные матрицы
    """
    values = np.array(data_agg.select([dat]).to_pandas()[[dat]])
    rows = np.array(data_agg.select([user]).to_pandas()[user].map(usr_dict))
    cols = np.array(data_agg.select([url]).to_pandas()[url].map(url_dict))
    mat = scipy.sparse.coo_matrix(
        (values.reshape((len(values))), (rows, cols)),
        shape=((rows.max() + 1), (cols.max() + 1).astype("int64")),
    )
    return mat


def pipeline_preprocess(data: pd.DataFrame, flg_evaluate: bool = True, **kwargs):
    """
    Пайплайн по предобработке данных
    :param data: датасет
    :return: датасет
    """
    # datamerge
    data = data_merge(data_1=data, data_2=pd.read_parquet(kwargs["target_train_path"]))

    # filling pass
    filling_pass(data=data, list_non=kwargs["list_non"])

    # replace values
    replace_values(data=data, map_change_columns=kwargs["map_change_columns"])

    # get bins
    for key in kwargs["map_bins_columns"].keys():
        if key == "day":
            data["day"] = pd.to_datetime(data.date).dt.day
            data[f"bins_{key}"] = data[key].apply(
                lambda x: get_bins(
                    x,
                    first_val=kwargs["map_bins_columns"][key][0],
                    second_val=kwargs["map_bins_columns"][key][1],
                )
            )
        else:
            data[f"bins_{key}"] = data[key].apply(
                lambda x: get_bins(
                    x,
                    first_val=kwargs["map_bins_columns"][key][0],
                    second_val=kwargs["map_bins_columns"][key][1],
                )
            )

    # data drop
    data_drop(data=data, drop_columns=kwargs["drop_columns"])

    # transform types
    transform_types(data=data, change_type_columns=kwargs["change_type_columns_test"])

    # data agg
    data_agg = data_aggregate(
        data=data, list_select=kwargs["list_select"], list_group=kwargs["list_group"]
    )

    # создает словари url и user
    url_set = set(data_agg.select(["url_host"]).to_pandas()["url_host"])
    url_dict = {url: idurl for url, idurl in zip(url_set, range(len(url_set)))}
    usr_set = set(data_agg.select(["user_id"]).to_pandas()["user_id"])
    usr_dict = {usr: user_id for usr, user_id in zip(usr_set, range(len(usr_set)))}

    # gen mat
    mat_1 = gen_mat(
        data_agg=data_agg,
        url_dict=url_dict,
        usr_dict=usr_dict,
        dat="month_count_distinct",
        user="user_id",
        url="url_host",
    )
    mat_2 = gen_mat(
        data_agg=data_agg,
        url_dict=url_dict,
        usr_dict=usr_dict,
        dat="city_name_count_distinct",
        user="user_id",
        url="url_host",
    )
    mat_3 = gen_mat(
        data_agg=data_agg,
        url_dict=url_dict,
        usr_dict=usr_dict,
        dat="cpe_model_name_count_distinct",
        user="user_id",
        url="url_host",
    )
    mat_4 = gen_mat(
        data_agg=data_agg,
        url_dict=url_dict,
        usr_dict=usr_dict,
        dat="bins_price_count_distinct",
        user="user_id",
        url="url_host",
    )
    mat_5 = gen_mat(
        data_agg=data_agg,
        url_dict=url_dict,
        usr_dict=usr_dict,
        dat="part_of_day_count_distinct",
        user="user_id",
        url="url_host",
    )
    mat_6 = gen_mat(
        data_agg=data_agg,
        url_dict=url_dict,
        usr_dict=usr_dict,
        dat="bins_age_count_distinct",
        user="user_id",
        url="url_host",
    )

    fin_mat = scipy.sparse.hstack((mat_1, mat_2, mat_3, mat_4, mat_5, mat_6)).tocsr()

    als = implicit.als.AlternatingLeastSquares(
        factors=kwargs["factors"],
        iterations=kwargs["iterations"],
        use_gpu=False,
        calculate_training_loss=False,
        regularization=kwargs["regularization"],
        random_state=kwargs["random_state"],
    )
    als.fit(fin_mat)

    u_factors = als.user_factors
    inv_usr_map = {v: k for k, v in usr_dict.items()}
    usr_emb = pd.DataFrame(u_factors)
    usr_emb["user_id"] = usr_emb.index.map(inv_usr_map)
    df_test = pd.read_parquet(kwargs["target_train_path"]).merge(
        usr_emb, how="inner", on=["user_id"]
    )
    df_test = df_test[df_test[kwargs['target_column']] != 'NA']
    df_test = df_test.dropna()
    df_test[kwargs['target_column']] = df_test[kwargs['target_column']].astype('int8')

    return df_test
