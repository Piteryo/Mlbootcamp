from data_processing import process_data
from LogisticRegression import LogisticRegressionPredictor
from CatboostPredictor import CatboostPredictor

import argparse

models = {'CatBoost Classifier': CatboostPredictor(
              {
                  "iterations": 500,
                  "depth": 3,
                  "learning_rate": 0.01,
                  "l2_leaf_reg": 0.3,
                  "rsm": 0.7,
                  "scale_pos_weight": 4,
                  "loss_function": 'Logloss',
                  "eval_metric": 'AUC',
                  "od_pval": 1e-5
              }
          ),
          'Logistic Regression': LogisticRegressionPredictor({'class_weight': 'balanced'})}

def main(trainPath, testPath, submissionPath, processData=True, X_train=None, X_test=None, y_train=None, y_test=None, X_submission_df=None, X_submission=None):
    max_score = 0
    iter = 0
    if processData:
        X_train, X_test, y_train, y_test, X_submission_df, X_submission = process_data(trainPath, testPath)
    for description, model in models.items():
        print(description)
        print(model.fitModel(X_train, y_train))

        score = model.getTestScore(X_test, y_test)

        if score > max_score:
            max_score = score

        print(score)
        model.evaluateModel(X_submission, X_submission_df, submissionPath, iter)
        iter += 1
    return max_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process paths.')
    parser.add_argument('paths', metavar='P', type=str, nargs="+",
                        help='paths to train, test folders and path to save submissions')
    args = parser.parse_args()

    main(*args.paths)
