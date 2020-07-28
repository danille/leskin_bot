from unittest.mock import Mock, patch

import numpy as np

from src.recognition.classifier.strategy import CNNImageClassificationStrategy


class TestClassificationStrategy:

    def test_cnn_classification_strategy_classify(self, model_mock, graph_mock):
        expected_prediction = np.array([.1, .2, .3])
        model_mock.predict = Mock(return_value=expected_prediction)

        strategy = CNNImageClassificationStrategy(model_mock, graph_mock)
        prediction = strategy.run(np.array([0.1, 0.1, 0.1]))

        assert (prediction == expected_prediction).all()

    @patch("src.recognition.classifier.strategy.load_model_from_filesystem", Mock(return_value=Mock()))
    def test_crete_cnn_strategy(self):
        config = {"model_name": "foo"}
        strategy = CNNImageClassificationStrategy.create(config)

        assert strategy.model is not None
        assert strategy.graph is not None
