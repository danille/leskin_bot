"""
This modules contains implementation of ImageClassifier
"""
from typing import List

import numpy as np

from src.recognition.classifier.strategy import ImageClassificationStrategy


class ImageClassifier:
    """
    Image classifier which classifies image
    using provided ImageClassificationStrategy
    """

    def __init__(self, image_classification_strategy: ImageClassificationStrategy, classes: List[str]):
        self._classification_strategy = image_classification_strategy
        self.classes = classes

    def classify(self, image: np.ndarray) -> str:
        """
        Classify image using provided strategy
        and returns name of the most probable class
        :param image: image which should be classified
        :return: name of the most probable class
        """
        # TODO: add threshold of classification
            # TODO: make strategy return the whole prediction array
        class_index = self._classification_strategy.run(image)

        return self.classes[class_index]
