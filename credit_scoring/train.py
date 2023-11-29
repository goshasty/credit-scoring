from lightgbm import LGBMClassifier

from credit_scoring.preprocessing import GetX

from .utils import save_model


class LGBMFit:
    def __init__(self, path_fitted_model):
        self.params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            # "feature_fraction": 0.9,
            "n_estimators": 5,  # 100
        }

        self.y_col = "target"
        self.ts_col = "ts"
        self.tech_cols = [self.y_col, self.ts_col]

        self.path_fitted_model = path_fitted_model

    def __fit_boosting(self, X_train, y_train, params):
        lgb_classifier = LGBMClassifier(**self.params, verbose=-1)
        lgb_classifier.fit(
            X_train,
            y_train
            # eval_set=[(X_test_pca, y_test)],
            # callbacks=[early_stopping(10)],
        )

        return lgb_classifier

    def fit_boosting(self, train_data, params=None):
        params = params if params is not None else self.params
        X_train = GetX(self.tech_cols).fit_transform(train_data)
        y_train = train_data[self.y_col]

        return self.__fit_boosting(X_train, y_train, params)

    def save_model(self, lgb_classifier):
        save_model(lgb_classifier, self.path_fitted_model)
