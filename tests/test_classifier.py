from unittest.mock import Mock

import numpy as np

from src.recognition.classifier import ImageClassifier


class TestClassifier:
    def test_classifier_above_threshold(self, classification_strategy_mock):
        classifier = ImageClassifier(classification_strategy_mock, ["foo", "bar", "test"], "default", 0.3)
        expected_result = "test"
        result = classifier.classify(np.array([1, 1, 1]))

        assert result == expected_result

    def test_classifier_under_threshold(self, classification_strategy_mock):
        classifier = ImageClassifier(classification_strategy_mock, ["foo", "bar", "test"], "default", 0.4)
        expected_result = "default"
        result = classifier.classify(np.array([1, 1, 1]))

        assert result == expected_result

