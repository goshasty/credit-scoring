import logging

import fire

from credit_scoring.infer import LGBMInfer
from credit_scoring.train import LGBMFit
from credit_scoring.utils import load_data, logging_setup, save_predicts


@logging_setup
def train(
    path_train_data: str = "data/train.csv",
    path_fitted_model: str = "models/v1.pickle",
):
    logging.info(path_train_data)
    logging.info(path_fitted_model)

    train_data = load_data(path_train_data)
    fitter = LGBMFit(path_fitted_model)

    logging.info("Start fitting")
    lgb_classifier = fitter.fit_boosting(train_data)
    fitter.save_model(lgb_classifier)

    logging.info("Model was saved")


@logging_setup
def infer(
    path_infer_data: str = "data/X_test.csv",
    path_save_predicts: str = "data/test_probas.csv",
    path_fitted_model: str = "models/v1.pickle",
):
    logging.info(path_fitted_model)

    infer_data = load_data(path_infer_data)
    infer = LGBMInfer(path_fitted_model)

    logging.info("Start predicting")
    predicts = infer.predict_proba(infer_data, 1)

    save_predicts(predicts, path_save_predicts)


if __name__ == "__main__":
    fire.Fire()
