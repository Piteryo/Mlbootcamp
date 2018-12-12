from data_processing import process_data
from LogisticRegression import LogisticRegressionPredictor
from CatboostPredictor import CatboostPredictor

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
          'Logistic Regression': LogisticRegressionPredictor({'class_weight': 'balanced'}),}

def main():
    X_train, X_test, y_train, y_test, X_submission_df, X_submission = process_data()
    for description, model in models.items():
        print(description)
        print(model.fitModel(X_train, y_train))
        print(model.getTestScore(X_test, y_test))
        model.evaluateModel(X_submission, X_submission_df)

if __name__ == "__main__":
    main()