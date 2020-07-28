import numpy as np

from src.recognition.classifier import ImageClassifier
from src.recognition.classifier.strategy import CNNImageClassificationStrategy
from src.recognition.cropper.base import ImageCropper
from src.recognition.cropper.strategy import BasicImageCropStrategy
from configparser import ConfigParser

CLASSIFIER_CONFIG_SECTION_NAME = 'recognition'

CLASSIFICATION_STRATEGY_MAP = {CNNImageClassificationStrategy.NAME: CNNImageClassificationStrategy}
CROP_STRATEGY_MAP = {BasicImageCropStrategy.NAME: BasicImageCropStrategy}


class RecognitionEngine:
    def __init__(self, image_classifier: ImageClassifier, image_cropper: ImageCropper):
        self.image_classifier = image_classifier
        self.image_cropper = image_cropper

    def classify(self, image: np.ndarray) -> str:
        """
        Classify raw user image and return name of the predicted class.
        :param image: raw user image
        :return: name of the predicted class
        """
        cropped_image = self.image_cropper.crop(image)
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

        cropper = ImageCropper(CROP_STRATEGY_MAP[crop_strategy_name]())
        classifier = ImageClassifier(CLASSIFICATION_STRATEGY_MAP[classification_strategy_name].create(), list(range(10)))

        return cls(classifier, cropper)
