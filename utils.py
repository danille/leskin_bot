import typing
import yaml
import os

# from google.cloud import storage

from keras.models import load_model

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_NAME = os.environ["MODEL_NAME"]


def read_yaml(path2yml: str) -> typing.Dict:
    with open(path2yml, mode='r') as yml_file:
        yaml_contents = yaml.safe_load(yml_file)
    return yaml_contents


# def download_blob(bucket_name, source_blob_name, destination_file_name):
#     project_name = os.environ["PROJECT_NAME"]
#     """Downloads a blob from the bucket."""
#     # credentials = app_engine.Credentials()
#     storage_client = storage.Client()
#
#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob(source_blob_name)
#     blob.download_to_filename(destination_file_name)
#
#     print(
#         "Blob {} downloaded to {}.".format(
#             source_blob_name, destination_file_name
#         )
#     )


def load_model_from_filesystem():
    # default_bucket_name = "leskin-bot.appspot.com"
    # if os.environ["LB_DEBUG"] == "1":
    path_to_model = os.path.join(MODELS_DIR, MODEL_NAME)
    # else:
    #     download_blob(default_bucket_name, MODEL_NAME, MODEL_NAME)
    #     path_to_model = MODEL_NAME
    model = load_model(path_to_model)
    return model
