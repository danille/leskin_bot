"""
This module contains implementation of ImageCropStrategy base class.
All other ImageCropStrategy should inherit from this class.
"""

import abc

import numpy as np


class ImageCropStrategy(abc.ABC):
    """
    Basic class for image crop strategies
    """
    def run(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        ...


class BasicImageCropStrategy(ImageCropStrategy):
    """
    Implementation of basic image crop strategy.py,
    which crops image around it's center.
    """
    NAME = "basic"

    def __init__(self):
        pass

    def run(self, image: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Crop image center
        :param image: ND array with pixel values
        :param new_height: height image should be cropped to
        :param new_width: width image should be cropped to
        :return: [new_height, new_width, channels] ND array with pixels values
        from center of original image
        """
        new_height = kwargs["new_height"]
        new_width = kwargs["new_width"]

        image.representation = None
        width, height, channels = image.representation.shape  # Get dimensions

        left = int(np.ceil((width - new_width) / 2))
        right = width - int(np.floor((width - new_width) / 2))

        top = int(np.ceil((height - new_height) / 2))
        bottom = height - int(np.floor((height - new_height) / 2))

        center_cropped_img = image.representation[top:bottom, left:right]
        return center_cropped_img
