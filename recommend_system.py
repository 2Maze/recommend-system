import numpy as np

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

def sorting(id_cat, vector):
    """
    Функция сортирует новости учитывая интересы пользователя
    :param id_cat: Словарь категорий
    :param vector: Вектор с чем будем сравнивать
    :return: Сортированный массив id новостей
    """
    metrics_dict = dict()
    news_dict = {}

    for cl in id_cat.items():
        news_dict[cl[0]] = [cl[1].TableGames,
                            cl[1].MusicConcert,
                            cl[1].Ballet,
                            cl[1].Opera,
                            cl[1].Theatre,
                            cl[1].Hackaton,
                            cl[1].Cinema,
                            cl[1].Vebinar,
                            cl[1].CyberSport,
                            cl[1].Festival]


    for id_news in news_dict:
        metrics_dict[id_news] = cosine_similarity(news_dict[id_news], vector)

    sorted_values = sorted(metrics_dict.values(), reverse=True)
    sorted_dict = {}

    for i in sorted_values:
        for k in metrics_dict.keys():
            if metrics_dict[k] == i:
                sorted_dict[k] = metrics_dict[k]
                metrics_dict.pop(k)
                break

    return sorted_dict.keys()
