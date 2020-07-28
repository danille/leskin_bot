"""
This modules contains implementation of ImageClassifier
"""
from typing import List, Union

import numpy as np

from src.recognition.classifier.strategy import ImageClassificationStrategy


class ImageClassifier:
    """
    Image classifier which classifies image
    using provided ImageClassificationStrategy
    """

    def __init__(self,
                 image_classification_strategy: ImageClassificationStrategy,
                 classes: Union[List[str], List[int]],
                 default_class: Union[str, int],
                 classification_threshold: float):
        self._classification_threshold = classification_threshold
        self._classification_strategy = image_classification_strategy
        self.classes = classes
        self.default_class = default_class

    def classify(self, image: np.ndarray) -> Union[str, int]:
        """
        Classify image using provided strategy
        and returns name of the most probable class
        :param image: image which should be classified
        :return: name of the most probable class
        """
        prediction = self._classification_strategy.run(image)
        class_index = int(np.argmax(prediction))
        if prediction[class_index] >= self._classification_threshold:
            return self.classes[class_index]
        else:
            return self.default_class
