from typing import List

from omegaconf import OmegaConf


class LGBM:
    def __init__(self, lgbt_params: OmegaConf, cols: List[str], *args):
        self.params = lgbt_params
        self.cols = cols
        self.tech_cols = [col for _, col in self.cols.items()]
