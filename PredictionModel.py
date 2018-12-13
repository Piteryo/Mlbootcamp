from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

import numpy as np

import os

class PredictionModel:
    def __init__(self, params):
        self.model = LogisticRegression()

    def getTestScore(self, X_test, y_test):
        return roc_auc_score(y_test, self.model.predict_proba(X_test)[:,1])

    def fitModel(self, X_train, y_train):
        raise NotImplementedError("Not implemented in base class.")

    def evaluateModel(self, X_submission, X_submission_df, submission_path, iter):
        res = self.model.predict_proba(X_submission)
        res = res[:, 1]
        X_submission_df["pred"] = res

        X_submission_df = X_submission_df["pred"].groupby(by="SK_ID").agg({'pred': np.mean})

        X_submission_df = X_submission_df.values

        np.savetxt(os.path.join(submission_path, "submission" + iter + ".csv"), X_submission_df.T[0], delimiter="\n")
        print("Saved to submission" + iter + ".csv")