from unittest.mock import Mock

import numpy as np

from src.recognition.preproccesor.base import ImagePreprocessor


class TestPreprocessor:
    def test_preprocessor(self):
        image = np.array([1, 1, 1])
        preprocess_strategy = Mock()
        preprocess_strategy.run = Mock(return_value=image)

        preprocessor = ImagePreprocessor(preprocess_strategy)

        result = preprocessor.preprocess(image)

        assert (result == image).all()
