"""
Программа: Отрисовка графиков
Версия: 1.0
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def barplot_group(col_main: str, col_group: str, data: pd.DataFrame,
                  title: str, grad: int) -> matplotlib.figure.Figure:
    """
    Строит barplot с нормированными данными с выводом значений на графике
    :param col_main: признак для анализа по col_group
    :param col_group: признак для нормализации/группировки
    :param data: датасет
    :param title: название графика
    :param grad: угол повората текста
    :return: поле рисунка
    """
    fig = plt.figure(figsize=(40, 15))

    data = (data.groupby([col_group])[col_main]
            .value_counts(normalize=True)
            .rename('Проценты')
            .mul(100)
            .reset_index()
            .sort_values(col_group))

    ax = sns.barplot(x=col_main, y="Проценты", hue=col_group, data=data)

    for p in ax.patches:
        percentage = '{:.1f}%'.format(p.get_height())
        ax.annotate(percentage,
                    (p.get_x() + p.get_width() / 2.,
                     p.get_height()),
                    ha='center',
                    va='center',
                    xytext=(0, 7),
                    textcoords='offset points',
                    fontsize=8)

    plt.title(title, fontsize=16)
    plt.ylabel('Проценты', fontsize=14)
    plt.xlabel(col_main, fontsize=14)
    plt.xticks(rotation=grad)
    return fig


def boxplot_group(targ_main: str, data: pd.DataFrame,
                    exp_group: str, title: str) -> matplotlib.figure.Figure:
    """
    Строит boxplot
    :param targ_main: целевой признак
    :param data: датасет
    :param exp_group: признак для анализа
    :param title: название графика
    :return: поле рисунка
    """
    fig = plt.figure(figsize=(40, 15))

    sns.boxplot(y=targ_main, x=exp_group, data=data, orient='h')

    plt.title(title, fontsize=18)
    plt.ylabel(targ_main, fontsize=15)
    plt.xlabel(exp_group, fontsize=15)
    return fig
