import numpy as np

from src.recognition.preproccesor.strategy import BasicImagePreprocessStrategy


class TestPreprocessStrategy:
    def test_basic_processor_strategy(self):
        test_image = np.array([[[1, 1, 1, 1],
                                [2, 2, 2, 2],
                                [3, 3, 3, 3],
                                [4, 4, 4, 4]],
                               [[1, 1, 1, 1],
                                [2, 2, 2, 2],
                                [3, 3, 3, 3],
                                [4, 4, 4, 4]]
                               ]).T
        expected_image = np.array([[[2, 2],
                                    [3, 3]],
                                   [[2, 2],
                                    [3, 3]]])

        strategy = BasicImagePreprocessStrategy(2, 2)
        image = strategy.run(test_image)

        assert (image == expected_image).all()

    def test_create_strategy(self):
        test_value = 1
        config = {"width": test_value, "height": test_value}
        strategy = BasicImagePreprocessStrategy.create(config)
        assert strategy.width == test_value
        assert strategy.height == test_value
