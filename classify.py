import time

import numpy as np
from keras.models import load_model

model = load_model('full_skin_cancer_model.h5')


def classify(image_array: np.ndarray) -> np.ndarray:
    """
    This function allows you to classify image data array
    :param image_array: NumPy array containing raw pixel values in RGB.
    Shape: [1, 299, 299, 3] as training images shape
    :return: NumPy array with probabilities for each class.
    Shape: [1, 7] as number of classes
    """
    prediction = model.predict(image_array)
    return prediction


def prediction2class(prediction: np.ndarray, classes: list = None) -> int:
    """
    This function allows you to get class from predicted probabilities
    :param prediction: NumPy array with class probability values
    :param classes: list of classes. If not specified,
    index of a class with highest probability will be return.
    :return: if classes is not specified, index of a class with the highest
    probability will be returned. Otherwise, value from classes will be returned
    which is located under highest probability index
    """
    most_probable_class_index = np.argmax(prediction)
    if classes:
        return classes[most_probable_class_index]
    else:
        return most_probable_class_index


if __name__ == '__main__':
    from keras.preprocessing.image import load_img, img_to_array

    image = load_img(
        "/Users/dlebediev/workspace/leskin_bot/data/skin-cancer-mnist-ham10000/HAM10000_images/ISIC_0034297.jpg",
        target_size=(299, 299))
    image_arr = img_to_array(image)
    image_reshaped = image_arr.reshape((1, *image_arr.shape))
    start_time = time.time()
    prediction = classify(image_reshaped)
    predicted_class = prediction2class(prediction)
    print(f"Class: {predicted_class}. Time to classify: {time.time() - start_time}")
