import logging

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from credit_scoring.preprocessing import GetX

from .utils import load_model


class LGBMInfer:
    def __init__(self, path_model_to_use):
        self.path_model_to_use = path_model_to_use
        self.model = self.load_model()

        self.y_col = "target"
        self.ts_col = "ts"
        self.tech_cols = [self.y_col, self.ts_col]

    def predict_proba(
        self, X_test: pd.DataFrame, target_class=None
    ) -> np.ndarray:
        X_test = GetX(self.tech_cols).fit_transform(X_test)
        propability = self.model.predict_proba(X_test)

        logging.info(f"Len of data is {len(propability)}")

        if target_class is None:
            return propability
        elif target_class == 0:
            return propability[:, 0]
        elif target_class == 1:
            return propability[:, 1]
        else:
            raise ValueError(
                "target_class parametr should ve one of {0, 1, None}"
            )

        """
        match target_class:
            case None:
                return propability
            case 1:
                return propability[:, 1]
            case 0:
                return propability[:, 0]
            case _:
                raise ValueError(
                    "target_class parametr should ve one of {0, 1, None}"
                )
        """

    def load_model(self) -> LGBMClassifier:
        lgb_classifier = load_model(self.path_model_to_use)
        return lgb_classifier
