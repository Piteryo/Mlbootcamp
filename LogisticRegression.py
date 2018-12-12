from sklearn.linear_model import LogisticRegressionCV
from PredictionModel import PredictionModel


class LogisticRegressionPredictor(PredictionModel):
    def __init__(self, params):
        self.model = LogisticRegressionCV(**params)

    def fitModel(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model.scores_
