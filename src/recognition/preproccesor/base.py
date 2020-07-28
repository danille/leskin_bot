"""
This module contains implementation of ImageCropper.
All extensions of ImageCropper should inherit from this class
"""
import numpy as np

from src.recognition.preproccesor.strategy import ImagePreprocessStrategy


class ImagePreprocessor:
    def __init__(self, image_crop_strategy: ImagePreprocessStrategy):
        self._crop_strategy = image_crop_strategy

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return self._crop_strategy.run(image)
