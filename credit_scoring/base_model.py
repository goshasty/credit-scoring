from typing import List

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from credit_scoring.preprocessing import GetX


class LGBM:
    def __init__(self, lgbt_params: OmegaConf, cols: List[str], *args):
        self.params = lgbt_params
        self.cols = cols
        self.tech_cols = [col for _, col in self.cols.items()]

    def predict_proba(
        self, infer_data: pd.DataFrame, target_class=None
    ) -> np.ndarray:
        X_infer = GetX(self.tech_cols).fit_transform(infer_data)
        propability = self.model.predict_proba(X_infer)

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
