from configparser import ConfigParser

import numpy as np

from src.recognition.classifier import ImageClassifier
from src.recognition.classifier.strategy import CNNImageClassificationStrategy
from src.recognition.preproccesor.base import ImagePreprocessor
from src.recognition.preproccesor.strategy import BasicImagePreprocessStrategy

CLASSIFIER_CONFIG_SECTION_NAME = 'recognition'

CLASSIFICATION_STRATEGY_MAP = {CNNImageClassificationStrategy.NAME: CNNImageClassificationStrategy}
PREPROCESS_STRATEGY_MAP = {BasicImagePreprocessStrategy.NAME: BasicImagePreprocessStrategy}


class RecognitionEngine:
    def __init__(self, image_classifier: ImageClassifier, image_preprocessor: ImagePreprocessor):
        self.image_classifier = image_classifier
        self.image_preprocessor = image_preprocessor

    def classify(self, image: np.ndarray) -> str:
        """
        Classify raw user image and return name of the predicted class.
        :param image: raw user image
        :return: name of the predicted class
        """
        cropped_image = self.image_preprocessor.preprocess(image)
        predicted_class = self.image_classifier.classify(cropped_image)
        return predicted_class

    @classmethod
    def assemble(cls, config_path):
        config = ConfigParser()
        config.read(config_path)

        try:
            classifier_config = config[CLASSIFIER_CONFIG_SECTION_NAME]
        except KeyError:
            raise Exception(f"Can not read config for Recognition Engine. "
                            f"Please, make sure that {config_path} contains config for Recognition Engine.")

        crop_strategy_name = classifier_config["crop_strategy"]
        classification_strategy_name = classifier_config["classification_strategy"]

        cropper = ImagePreprocessor(PREPROCESS_STRATEGY_MAP[crop_strategy_name].create())
        classifier = ImageClassifier(CLASSIFICATION_STRATEGY_MAP[classification_strategy_name].create(),
                                     list(range(10)))

        return cls(classifier, cropper)
