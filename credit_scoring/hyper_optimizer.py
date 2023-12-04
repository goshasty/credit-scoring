import logging
from typing import Any, Dict

import mlflow
import pandas as pd
from hyperopt import fmin, hp, tpe
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from credit_scoring.preprocessing import Gety
from credit_scoring.train import LGBMFit


class Hyperopt:
    def __init__(
        self,
        cols: Dict[str, str],
        mlflow_server: str = "http://127.0.0.1:8080",
        max_evals: int = 10,
        exp_name: str = "exp",
        pr_rec_threshold: float = 0.1,
    ):
        self.space = {
            "learning_rate": hp.uniform("learning_rate", 0.01, 0.1),
            "n_estimators": hp.choice("n_estimators", range(50, 200)),
            "max_depth": hp.choice("max_depth", range(3, 12)),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
            "subsample": hp.uniform("subsample", 0.5, 1.0),
            "reg_alpha": hp.uniform("reg_alpha", 0, 1),
        }
        loggers_hyperopt = [
            logging.getLogger(name)
            for name in logging.Logger.manager.loggerDict
            if "hyperopt" in name
        ]
        for logger in loggers_hyperopt:
            logger.setLevel(logging.WARNING)

        self.mlflow_server = mlflow_server
        self.cols = cols
        self.max_evals = max_evals
        self.exp_name = exp_name
        self.pr_rec_threshold = pr_rec_threshold

    def optimize(self, train_data: pd.DataFrame) -> None:
        mlflow.set_tracking_uri(self.mlflow_server)

        mlflow.set_experiment(self.exp_name)
        self.train_data, self.val_data = train_test_split(
            train_data, test_size=0.2, random_state=42, shuffle=False
        )

        self.y_train = Gety(self.cols).fit_transform(self.train_data)
        self.y_val = Gety(self.cols).fit_transform(self.val_data)

        self.num_run = 0
        best = fmin(
            fn=lambda params: self.objective(params),
            space=self.space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=None,
            verbose=1,
        )
        print(best)

    def objective(self, params: Dict[str, Any]) -> float:

        lgb_fitter = LGBMFit(None, params, self.cols, verbose=False)
        lgb_fitter.fit_boosting(self.train_data)

        y_val_pred = lgb_fitter.predict_proba(self.val_data)[:, 1]
        y_train_pred = lgb_fitter.predict_proba(self.train_data)[:, 1]

        auc_train = roc_auc_score(self.y_train, y_train_pred)
        auc_val = roc_auc_score(self.y_val, y_val_pred)

        f1_score_train = f1_score(
            self.y_train, (y_train_pred > self.pr_rec_threshold) * 1
        )
        f1_score_val = f1_score(
            self.y_val, (y_val_pred > self.pr_rec_threshold) * 1
        )

        self.num_run += 1
        with mlflow.start_run(run_name=str(self.num_run)):
            mlflow.log_params(params)
            mlflow.log_metric("auc_train", auc_train)
            mlflow.log_metric("auc_val", auc_val)

            mlflow.log_metric("f1_score_train", f1_score_train)
            mlflow.log_metric("f1_score_val", f1_score_val)

        return -auc_val
