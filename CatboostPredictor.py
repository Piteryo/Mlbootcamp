from PredictionModel import PredictionModel
from catboost import CatBoostClassifier

from catboost import Pool, cv

import numpy as np


class CatboostPredictor(PredictionModel):
    def __init__(self, params):
        self.model = CatBoostClassifier(**params)

    def fitModel(self, X_train, y_train):
        self.model.fit(X_train, y_train, verbose=True, cat_features=np.arange(381, 384))
        pool = Pool(X_train, y_train, cat_features=np.arange(381, 384))
        scores = cv(pool, self.model.get_params(), verbose=True)
        return scores
