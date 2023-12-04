# from numpy import ndarray
from typing import List

from sklearn.base import BaseEstimator, TransformerMixin


class GetX(BaseEstimator, TransformerMixin):
    """
    Transofrmer is for extracting features from dataset
    """

    def __init__(self, cols_to_drop: List[str]):
        self.cols_to_drop = cols_to_drop

    def fit(self, Xy):
        return self

    def transform(self, Xy):
        cols_in_drop = [col for col in self.cols_to_drop if col in Xy.columns]
        X = Xy.drop(columns=cols_in_drop)
        return X


class Gety(BaseEstimator, TransformerMixin):
    """
    Transofrmer is for extracting target
    """

    def __init__(self, cols: List[str]):
        self.y_col = cols["y_col"]

    def fit(self, Xy):
        return self

    def transform(self, Xy):
        y = Xy[self.y_col]
        return y
