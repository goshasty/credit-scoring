import logging
import pickle
from typing import Callable, Union

import dvc.api
import numpy as np
import onnxmltools
import pandas as pd
from lightgbm import LGBMClassifier
from onnxconverter_common.data_types import FloatTensorType


def load_data(path_data: str) -> pd.DataFrame:
    with dvc.api.open(path_data) as f:
        df = pd.read_csv(f)
    logging.info(f"Load train data from {path_data} with {len(df)} rows")
    return df


def save_model(
    model: LGBMClassifier, path_model: str, path_onnx_model: str
) -> None:
    with open(path_model, "wb") as model_file:
        pickle.dump(model, model_file)
    logging.info(f"Model  was saved: {path_model}")

    n_features = model.n_features_
    initial_types = [("input", FloatTensorType([None, n_features]))]
    onnx_model = onnxmltools.convert_lightgbm(
        model, initial_types=initial_types
    )
    onnxmltools.utils.save_model(onnx_model, path_onnx_model)
    logging.info(f"Model onnx  was saved: {path_onnx_model}")


def load_model(path_model: str) -> LGBMClassifier:
    with open(path_model, "rb") as model_file:
        model = pickle.load(model_file)
    return model


def save_predicts(
    predicts: Union[np.ndarray, pd.DataFrame], path_to_save: str
) -> None:
    if isinstance(predicts, np.ndarray):
        to_save = pd.DataFrame(data=predicts, columns=["propability"])
    elif isinstance(predicts, pd.DataFrame):
        to_save = predicts
    else:
        raise ValueError("must be pd or np")
    to_save.to_csv(path_to_save, index=False)
    logging.info(f"Predictions were saved {path_to_save}")


def logging_setup(f: Callable) -> Callable:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    def inner(*args, **kwargs):
        return f(*args, **kwargs)

    return inner
