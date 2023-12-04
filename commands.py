import logging
from typing import Dict, Union

import fire
import mlflow
import numpy as np
import onnxmltools
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from mlflow.models import infer_signature
from omegaconf import OmegaConf

from credit_scoring.config import Params
from credit_scoring.hyper_optimizer import Hyperopt
from credit_scoring.infer import LGBMInfer
from credit_scoring.train import LGBMFit
from credit_scoring.utils import load_data, logging_setup, save_predicts


def get_config(conf_path: str) -> OmegaConf:
    cs = ConfigStore.instance()
    cs.store(name="params", node=Params)
    with initialize(version_base="1.3", config_path=conf_path):
        config = compose(config_name="config")

    return config


@logging_setup
def train(config_path: str = "config") -> None:
    config = get_config(config_path)

    train_data = load_data(config.data.path_train_data)
    fitter = LGBMFit(
        config.model.path_fitted_model, config.model.params, config.data.cols
    )

    logging.info("Start fitting")
    lgb_classifier = fitter.fit_boosting(train_data)
    fitter.save_model(lgb_classifier)


@logging_setup
def infer(config_path: str = "config"):
    config = get_config(config_path)
    infer_data = load_data(config.data.path_infer_data)
    infer = LGBMInfer(
        config.model.path_fitted_model, config.model.params, config.data.cols
    )

    logging.info("Start predicting")
    predicts = infer.predict_proba(infer_data, 1)

    save_predicts(predicts, config.data.path_predicts)


@logging_setup
def hyperopt(config_path: str = "config"):
    config = get_config(config_path)
    train_data = load_data(config.data.path_train_data)
    hopt = Hyperopt(config.data.cols, config.mlflow.server, **config.hyperopt)
    best_params = hopt.optimize(train_data)
    logging.info(f"Best params: {repr(best_params)}")


@logging_setup
def run_server(
    input_data: Dict[str, Union[int, float]], config_path: str = "config"
):

    vals = [v for _, v in input_data.items()]
    print(input_data, vals)
    config = get_config(config_path)
    onnx_model = onnxmltools.utils.load_model(config.model.path_onnx_model)

    input_data = np.array([vals]).astype(float)

    print(input_data)

    with mlflow.start_run():
        signature = infer_signature(input_data, np.array([0.0]))
        model_info = mlflow.onnx.log_model(
            onnx_model, "model", signature=signature
        )

    onnx_pyfunc = mlflow.pyfunc.load_model(model_info.model_uri)
    predictions = onnx_pyfunc.predict(input_data)
    return predictions


if __name__ == "__main__":
    fire.Fire()
