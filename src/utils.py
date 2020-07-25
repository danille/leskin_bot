import typing
import yaml
import os

from keras.models import load_model

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_NAME = os.environ["MODEL_NAME"]


def read_yaml(path2yml: str) -> typing.Dict:
    with open(path2yml, mode='r') as yml_file:
        yaml_contents = yaml.safe_load(yml_file)
    return yaml_contents


def load_model_from_filesystem():
    path_to_model = os.path.join(MODELS_DIR, MODEL_NAME)
    model = load_model(path_to_model)
    return model
