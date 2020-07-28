"""
This module consists implementation of ImageClassificationStrategy.
All ImageClassificationStrategy'ies should inherit from it.
"""
import abc

import numpy as np
import tensorflow as tf

from src.utils import load_model_from_filesystem


class ImageClassificationStrategy(abc.ABC):
    """
    Base class for image classification strategies
    """
    def run(self, image: np.ndarray) -> int:
        """
        Abstract method for classification of the image
        :param image: image which should be classified
        :return: index of the most probable class
        """
        ...


class CNNImageClassificationStrategy(ImageClassificationStrategy):
    """
    Implementation of image classification strategy
    based on CNN model usage
    """
    NAME = "cnn"

    def __init__(self, model, graph):
        self.model = model
        self.graph = graph

    def run(self, image: np.ndarray) -> int:
        """
        Classify object on the image using the provided CNN model
        :param image: image which should be classified
        :return: an index of the most probable class
        """
        with self.graph.as_default():
            prediction = self.model.predict(image)

        return np.argmax(prediction)

    @classmethod
    def create(cls):
        model = load_model_from_filesystem()
        graph = tf.get_default_graph()
        return cls(model, graph)
