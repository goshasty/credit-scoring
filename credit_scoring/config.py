from dataclasses import dataclass
from typing import Dict


@dataclass
class Data:
    path_train_data: str
    path_infer_data: str
    path_predicts: str
    cols: Dict[str, str]


@dataclass
class BoostingParams:
    boosting_type: str
    learning_rate: float
    metric: str
    n_estimators: int
    num_leaves: int
    objective: str


@dataclass
class Model:
    name: str
    path_fitted_model: str
    path_onnx_model: str
    params: BoostingParams


@dataclass
class MLFlow:
    server: str


@dataclass
class Hyperopt:
    max_evals: int
    exp_name: str
    pr_rec_threshold: float


@dataclass
class Params:
    data: Data
    model: Model
    mlflow: MLFlow
    hyperopt: Hyperopt
