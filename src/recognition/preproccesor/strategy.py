"""
This module contains implementation of ImageCropStrategy base class.
All other ImageCropStrategy should inherit from this class.
"""

import abc

import numpy as np


class ImagePreprocessStrategy(abc.ABC):
    """
    Basic class for image crop strategies
    """

    def run(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        ...


class BasicImagePreprocessStrategy(ImagePreprocessStrategy):
    NAME = "basic"

    def __init__(self, width, height):
        """
        Implementation of basic image crop strategy.py,
        which crops image around it's center.
        :param width: width to which image should be cropped
        :param height: height to which image should be cropped
        """
        self.width = width
        self.height = height

    def run(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Crop image center
        :param image: ND array with pixel values
        :return: [new_height, new_width, channels] ND array with pixels values
        from center of original image
        """
        new_height = self.height
        new_width = self.width
        height, width, channels = image.shape  # Get dimensions

        left = int(np.ceil((width - new_width) / 2))
        right = width - int(np.floor((width - new_width) / 2))

        top = int(np.ceil((height - new_height) / 2))
        bottom = height - int(np.floor((height - new_height) / 2))

        center_cropped_img = image[top:bottom, left:right, :]
        return center_cropped_img

    @classmethod
    def assemble(cls, config):
        height = config["height"]
        width = config["width"]
        return cls(width, height)
