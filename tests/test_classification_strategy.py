from unittest.mock import Mock

import numpy as np

from src.recognition.classifier.strategy import CNNImageClassificationStrategy


class TestClassificationStrategy:

    def test_cnn_classification_strategy_classify(self, model_mock, graph_mock):
        expected_prediction = np.array([.1, .2, .3])
        model_mock.predict = Mock(return_value=expected_prediction)

        strategy = CNNImageClassificationStrategy(model_mock, graph_mock)
        prediction = strategy.run(np.array([0.1, 0.1, 0.1]))

        assert (prediction == expected_prediction).all()
