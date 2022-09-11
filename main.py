import sys
import numpy as np
import typing

def cosine_similarity(news_vector, user_vector):
    """
    Функция вычисляет косинус угла между векторами,
    Это можно интерпретировать как близость векторов в многомерном пространстве
    :param news_vector: Вектор новости
    :param user_vector: Вектор предпочтений
    :return: Метрику схожести двух векторов
    """
    n_v = np.array(news_vector, dtype=int)
    u_v = np.array(user_vector, dtype=int)
    return np.dot(n_v, u_v) / (np.linalg.norm(n_v) * np.linalg.norm(u_v))

def sorting_news(news_dict, user_vector):
    """
    Функция сортирует новости учитывая интересы пользователя
    :param News_dict: Словарь новостей
    :param user_vector: Пользовательский вектор предочтений
    :return: Сортированный массив id новостей
    """
    metrics_dict = dict()

    for id_news in news_dict:
        metrics_dict[id_news] = cosine_similarity(news_dict[id_news], user_vector)

    sorted_values = sorted(metrics_dict.values(), reverse=True)
    sorted_dict = {}

    for i in sorted_values:
        for k in metrics_dict.keys():
            if metrics_dict[k] == i:
                sorted_dict[k] = metrics_dict[k]
                metrics_dict.pop(k)
                break

    print(sorted_dict)

if __name__ == "__main__":
    user = [0, 1, 1]
    news = dict({"12": [1, 1, 1],
                 "21": [0, 1, 0],
                 "33": [1, 1, 1],
                 "55": [1, 1, 1],
                 "10": [0, 0, 1]})

    # Здесь функция на проверку входных данных

    sorting_news(news, user)
