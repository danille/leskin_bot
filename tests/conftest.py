from unittest.mock import Mock

import numpy as np
import pytest


@pytest.fixture
def model_mock():
    return Mock()


@pytest.fixture
def graph_mock():
    graph = Mock()
    graph.as_default = Mock()
    graph.as_default.return_value.__enter__ = Mock(return_value="foo")
    graph.as_default.return_value.__exit__ = Mock(return_value=False)

    return graph


@pytest.fixture
def classification_strategy_mock():
    classification_strategy = Mock()
    classification_strategy.run = Mock(return_value=np.array([.1, .2, .3]))
    return classification_strategy
