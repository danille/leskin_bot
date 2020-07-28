from unittest.mock import Mock

import numpy as np

from src.recognition.classifier import ImageClassifier


class TestClassifier:
    def test_classifier(self):
        classification_strategy = Mock()
        classification_strategy.run = Mock(return_value=np.array([.1, .2, .3]))
        classifier = ImageClassifier(classification_strategy, ["foo", "bar", "test"])
        expected_result = "test"
        result = classifier.classify(np.array([1, 1, 1]))

        assert result == expected_result