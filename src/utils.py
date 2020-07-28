import typing
import yaml
import os

from keras.models import load_model

SRC_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")
# MODEL_NAME = os.environ["MODEL_NAME"]
CONFIG_PATH = os.path.join(BASE_DIR, "application.cfg")


def read_yaml(path2yml: str) -> typing.Dict:
    with open(path2yml, mode='r') as yml_file:
        yaml_contents = yaml.safe_load(yml_file)
    return yaml_contents


def load_model_from_filesystem(model_name):
    path_to_model = os.path.join(MODELS_DIR, model_name)
    model = load_model(path_to_model)
    return model
