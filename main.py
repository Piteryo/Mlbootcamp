from data_processing import process_data
from LogisticRegression import LogisticRegressionPredictor

models = [LogisticRegressionPredictor({'class_weight': 'balanced'})]

def main():
    X_train, X_test, y_train, y_test, X_submission_df, X_submission = process_data()
    for model in models:
        print(model.fitModel(X_train, y_train))
        print(model.getTestScore(X_test, y_test))
        model.evaluateModel(X_submission, X_submission_df)

if __name__ == "__main__":
    main()