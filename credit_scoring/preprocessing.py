# from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin


class GetX(BaseEstimator, TransformerMixin):
    """
    Transofrmer is for extracting features from dataset
    """

    def __init__(self, cols_to_drop):
        self.cols_to_drop = cols_to_drop

    def fit(self, Xy):
        return self

    def transform(self, Xy):
        cols_in_drop = [col for col in self.cols_to_drop if col in Xy.columns]
        X = Xy.drop(columns=cols_in_drop)
        return X
