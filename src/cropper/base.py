"""
This module contains implementation of ImageCropper.
All extensions of ImageCropper should inherit from this class
"""
import numpy as np

from src.cropper import ImageCropStrategy


class ImageCropper:
    def __init__(self, image_crop_strategy: ImageCropStrategy):
        self._crop_strategy = image_crop_strategy

    def crop(self, image: np.ndarray) -> np.ndarray:
        return self._crop_strategy.run(image)
