import tensorflow as tf

from src.recognition.engine import RecognitionEngine
from src.utils import load_model_from_filesystem
from src.recognition.classifier.strategy import CNNImageClassificationStrategy

from src.recognition.preproccesor.strategy import BasicImagePreprocessStrategy
from src.utils import CONFIG_PATH


class Application:
    @classmethod
    def run(cls):
        recognition_engine = RecognitionEngine.assemble(CONFIG_PATH)
