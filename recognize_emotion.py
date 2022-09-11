import numpy as np
import cv2

#from tensorflow.keras.models import load_model

detector = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_alt2.xml")

mood_vectors = {"0": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                "1": [1, 1, 0, 0, 0, 1, 1, 0, 1, 0],
                "2": [1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
                "3": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                "4": [0, 0, 0, 0, 0, 0, 1, 1, 0, 1]}
def preprocess_image(bytes_array):
    """
    Из байетового массива собирает numpy массив
    :param bytes_array: Байтовый массив
    :return: Numpy матрица
    """
    return np.array([1, 2, 3])

def get_face(img):
    """
    Получает на вход изображение, на выходе вектор эмоции
    :param img: Numpy матрица
    :return: Изображение numpy
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray,
                                      scaleFactor=1.1,
                                      minNeighbors=5,
                                      minSize=(60, 60),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    return cv2.cvtColor(cv2.resize(img[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]], (48, 48)), cv2.COLOR_BGR2GRAY)

def predict_emotion(img):
    """
    Функция предсказывает вектор эмоций по изображению
    :param img: Numpy массив размером (48, 48)
    :return: Номер эмоции, где 0 - страшный, 1 - счастливый, 2 - нейтральный, 3 - грустный, 4 - восторг
    """
    #model = load_model("models/img_to_emotion.h5")
    #return model.predict(np.expand_dims(np.expand_dims(img, -1), 0))

def recognize(img):
    """
    Из байт массива преобразует в numpy array, находит лицо и пронозирует настроение
    :param img: Байт массив
    :return: Настроение
    """

    return predict_emotion(get_face(preprocess_image(img)))

#if __name__ == "__main__":
    #face = get_face(cv2.imread("/home/danil/Downloads/7a03a4f018363854eb6c2cf8a4ca7177.jpg")))
    #predict_emotion(face)