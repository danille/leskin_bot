from typing import Optional

import numpy as np
import tensorflow as tf
from src.utils import load_model_from_filesystem

GRAPH = tf.get_default_graph()
MODEL = load_model_from_filesystem()


def classify(image_array: np.ndarray) -> np.ndarray:
    """
    This function allows you to classify image data array
    :param image_array: NumPy array containing raw pixel values in RGB.
    Shape: [1, 299, 299, 3] as training images shape
    :return: NumPy array with probabilities for each class.
    Shape: [1, 7] as number of classes
    """
    with GRAPH.as_default():
        prediction = MODEL.predict(image_array)
    return prediction


def crop_from_center(img, new_height, new_width):
    """
    Crop image center 
    :param img: ND array with pixel values
    :param new_height: height image should be cropped to
    :param new_width: width image should be cropped to
    :return: [new_height, new_width, channels] ND array with pixels values 
    from center of original image  
    """
    width, height, channels = img.shape  # Get dimensions

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    center_cropped_img = img[top:bottom, left:right]
    return center_cropped_img


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


def get_cancer_class_from(image: np.ndarray) -> Optional[str]:
    _, input_height, input_width, _ = MODEL.input_shape
    cropped_image = crop_from_center(image, input_height, input_width)
    image_for_model = cropped_image.reshape((1, *cropped_image.shape))
    prediction = classify(image_for_model)

    if np.any(prediction > .5):
        return f"{prediction2class(prediction)}"
    else:
        return None
