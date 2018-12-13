from PredictionModel import PredictionModel
import lightgbm

import numpy as np


class LightGbmPredictor(PredictionModel):
    def __init__(self, params):
        self.model = lightgbm.LGBMClassifier(**params)

    def fitModel(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return "Scores will be on validation set"
