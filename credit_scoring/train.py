import logging

from lightgbm import LGBMClassifier

from credit_scoring.base_model import LGBM
from credit_scoring.preprocessing import GetX
from credit_scoring.utils import save_model


class LGBMFit(LGBM):
    def __init__(self, path_fitted_model, lgbt_params, cols, verbose=True):
        super().__init__(lgbt_params, cols)
        self.path_fitted_model = path_fitted_model
        self.verbose = verbose

    def __fit_boosting(self, X_train, y_train, params):
        lgb_classifier = LGBMClassifier(**self.params, verbose=-1)
        lgb_classifier.fit(X_train, y_train)

        return lgb_classifier

    def fit_boosting(self, train_data, params=None):
        params = params if params is not None else self.params
        X_train = GetX(self.tech_cols).fit_transform(train_data)
        if self.verbose:
            logging.info(f"Shape of train data: {X_train.shape}")
        y_train = train_data[self.cols["y_col"]]
        self.model = self.__fit_boosting(X_train, y_train, params)

        return self.model

    def save_model(self, lgb_classifier):
        save_model(lgb_classifier, self.path_fitted_model)
