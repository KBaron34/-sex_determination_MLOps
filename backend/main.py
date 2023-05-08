"""
Программа: Модель для прогнозирования того, будет ли пользователь принадлежать
к мужскому или женскому полу, в зависимости от его интернет активночти и личных данных
Версия: 1.0
"""

import warnings
import pandas as pd

import uvicorn
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from pydantic import BaseModel

import src.pipelines.pipeline
import src.evaluate.evaluate
import src.train.metrics

warnings.filterwarnings("ignore")

app = FastAPI()
CONFIG_PATH = '../config/params.yml'


class InsuranceCustomer(BaseModel):
    """
    Признаки для получения результатов модели
    """
    region_name: str
    city_name: str
    cpe_manufacturer_name: str
    cpe_model_name: str
    cpe_model_os_type: str
    cpe_type_cd: str
    url_host: str
    request_cnt: int
    price: int
    date: str
    part_of_day: str
    user_id: int


@app.get('/hello')
def welcome():
    """
    Hello
    :return: None
    """
    return {'message': 'Hello data Scientist!'}


@app.post('/train')
def training():
    """
    Обучает модель, логирует метрики
    """
    src.pipeline_training(config_path=CONFIG_PATH)
    metrics = src.load_metrics(config_path=CONFIG_PATH)
    return {'metrics': metrics}


@app.post("/predict")
def prediction(file: UploadFile = File(...)):
    """
    Предсказывает по данным из файла
    """
    result = src.pipeline_evaluate(config_path=CONFIG_PATH, data_path=file.file)
    assert isinstance(result, list), "Результат не соответствует типу list"
    return {"prediction": result[:5]}


@app.post("/predict_input")
def prediction_input(customer: InsuranceCustomer):
    """
    Предсказывает пол по введенным данным
    """
    features = [
        [
            customer.region_name,
            customer.city_name,
            customer.cpe_manufacturer_name,
            customer.cpe_model_name,
            customer.cpe_model_os_type,
            customer.cpe_type_cd,
            customer.url_host,
            customer.request_cnt,
            customer.price,
            customer.date,
            customer.part_of_day,
            customer.user_id
        ]
    ]

    cols = [
        'region_name',
        'city_name',
        'cpe_manufacturer_name',
        'cpe_model_name',
        'cpe_model_os_type',
        'cpe_type_cd',
        'url_host',
        'request_cnt',
        'price',
        'date',
        'part_of_day',
        'user_id'
    ]

    data = pd.DataFrame(features, columns=cols)
    predictions = src.pipeline_evaluate(config_path=CONFIG_PATH, dataset=data)[0]
    result = (
        {"Male"}
        if predictions == 1
        else {"Female"}
        if predictions == 0
        else "Error result"
    )
    return result


if __name__ == "__main__":
    # Запустите сервер, используя заданный хост и порт
    uvicorn.run(app, host="127.0.0.1", port=80)
