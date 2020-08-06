from configparser import ConfigParser

import numpy as np

from src.recognition.classifier import ImageClassifier
from src.recognition.classifier.strategy import CNNImageClassificationStrategy
from src.recognition.preproccesor.base import ImagePreprocessor
from src.recognition.preproccesor.strategy import BasicImagePreprocessStrategy

CLASSIFIER_CONFIG_SECTION_NAME = "classifier"
PREPROCESSOR_CONFIG_SECTION_NAME = "preprocessor"

CLASSIFICATION_STRATEGY_MAP = {CNNImageClassificationStrategy.NAME: CNNImageClassificationStrategy}
PREPROCESS_STRATEGY_MAP = {BasicImagePreprocessStrategy.NAME: BasicImagePreprocessStrategy}

CLASSES = ["melanocytic nevi",
           "melanoma",
           "benign keratosis-like lesion",
           "basal cell carcinoma",
           "actinic keratoses",
           "vascular lesion",
           "dermatofibroma"]


class ClassificationEngine:
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
        image_for_model = cropped_image.reshape((1, *cropped_image.shape))
        predicted_class = self.image_classifier.classify(image_for_model)
        return predicted_class

    @classmethod
    def assemble(cls, config_path):
        config = ConfigParser()
        config.read(config_path)

        preprocessor = cls._assemble_preprocessor(config)
        classifier = cls._assemble_classifier(config)
        return cls(classifier, preprocessor)

    @classmethod
    def _assemble_classifier(cls, config):
        try:
            classifier_config = config[CLASSIFIER_CONFIG_SECTION_NAME]
        except KeyError:
            raise Exception(f"Can not read config for classifier. "
                            f"Please, make sure that config contains {CLASSIFIER_CONFIG_SECTION_NAME} section.")

        classification_strategy_name = classifier_config["strategy"]
        classification_threshold = float(classifier_config["threshold"])
        classifier = ImageClassifier(
            CLASSIFICATION_STRATEGY_MAP[classification_strategy_name].create(classifier_config),
            CLASSES,
            "bening nevi",
            classification_threshold=classification_threshold)

        return classifier

    @classmethod
    def _assemble_preprocessor(cls, config):
        try:
            preprocessor_config = config[PREPROCESSOR_CONFIG_SECTION_NAME]
        except KeyError:
            raise Exception(f"Can not read config for preprocessor. "
                            f"Please, make sure that config contains {PREPROCESSOR_CONFIG_SECTION_NAME} section.")

        preprocess_strategy_name = preprocessor_config["strategy"]
        preprocessor = ImagePreprocessor(
            PREPROCESS_STRATEGY_MAP[preprocess_strategy_name].create(preprocessor_config))

        return preprocessor
