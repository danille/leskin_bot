from src.recognition.engine import ClassificationEngine

from src.utils import CONFIG_PATH


class Application:
    @classmethod
    def run(cls):
        recognition_engine = ClassificationEngine.assemble(CONFIG_PATH)
