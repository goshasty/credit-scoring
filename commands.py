import logging

import fire
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from credit_scoring.config import Params
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


if __name__ == "__main__":
    fire.Fire()
