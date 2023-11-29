import logging
import pickle

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


def load_data(path_data: str):
    return pd.read_csv(path_data)


def save_model(model: LGBMClassifier, path_model: str):
    with open(path_model, "wb") as model_file:
        pickle.dump(model, model_file)


def load_model(path_model: str) -> LGBMClassifier:
    with open(path_model, "rb") as model_file:
        model = pickle.load(model_file)
    return model


def save_predicts(predicts: np.ndarray, path_to_save: str):
    to_save = pd.DataFrame(data=predicts, columns=["propability"])
    to_save.to_csv(path_to_save, index=False)


def logging_setup(f):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    def inner(*args, **kwargs):
        return f(*args, **kwargs)

    return inner
