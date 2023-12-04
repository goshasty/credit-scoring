from lightgbm import LGBMClassifier

from credit_scoring.base_model import LGBM
from credit_scoring.utils import load_model


class LGBMInfer(LGBM):
    def __init__(self, path_model_to_use, lgbt_params, cols):
        super().__init__(lgbt_params, cols)
        self.path_model_to_use = path_model_to_use
        self.model = self.load_model()

    def load_model(self) -> LGBMClassifier:
        lgb_classifier = load_model(self.path_model_to_use)
        return lgb_classifier
