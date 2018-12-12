import numpy as np

import pandas as pd

from pandas_profiling import ProfileReport

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import QuantileTransformer, FunctionTransformer
from sklearn.preprocessing import RobustScaler


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(
                        np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(
                        np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(
                        np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(
                        np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(
                        np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(
                        np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(
        100 * (start_mem - end_mem) / start_mem))

    return df

def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True, sep=';')
    df = reduce_mem_usage(df)
    return df

def sortByDate(df, validateSet=False):
    df['SNAP_DATE'] = pd.to_datetime(df['SNAP_DATE'], format='%d.%m.%y')
    df['BASE_TYPE'] -= 1
    if validateSet:
        df = df.groupby(by="SK_ID", group_keys=False).apply(lambda grp: grp.nlargest(1, 'SNAP_DATE'))
    return df

def process_submission_data(classifier_pipeline):
    subs_csi_1 = pd.read_csv('home/spolezhaev/test/subs_csi_test.csv', index_col='SK_ID')

    subs_features_1 = pd.read_csv('home/spolezhaev/test/subs_features_test.csv', index_col='SK_ID')
    subs_bs_consumption1 = pd.read_csv('home/spolezhaev/test/subs_bs_consumption_test.csv')

    df1 = subs_csi_1.merge(subs_features_1, on="SK_ID")

    df1 = df1.merge(subs_bs_consumption1.groupby(by=["SK_ID", "MON"], as_index=False).sum().set_index('SK_ID'),
                    how='left', on='SK_ID')

    df1 = sortByDate(df1)
    X_submission = df1.drop(columns=["SNAP_DATE", "CONTACT_DATE", 'COM_CAT#24'])
    return X_submission, classifier_pipeline.fit_transform(X_submission)

def process_data():
    subs_csi = pd.read_csv('home/spolezhaev/train/subs_csi_train.csv', index_col='SK_ID')
    subs_features = pd.read_csv('home/spolezhaev/train/subs_features_train.csv', index_col='SK_ID')
    subs_bs_consumption = pd.read_csv('home/spolezhaev/train/subs_bs_consumption_train.csv')

    df = subs_csi.merge(subs_features, on="SK_ID")
    df = df.merge(subs_bs_consumption.groupby(by=["SK_ID", "MON"], as_index=False).sum().set_index('SK_ID'), on='SK_ID')
    df = sortByDate(df)
    df = df[df['ACT'] == 1]
    X, y = df.drop(columns=["CSI", "SNAP_DATE", "CONTACT_DATE", 'COM_CAT#24', 'ACT', 'MON', 'CELL_LAC_ID']), df["CSI"]

    profileReport = ProfileReport(df)
    categorical_features = ['ARPU_GROUP', 'DEVICE_TYPE_ID',
                            'INTERNET_TYPE_ID']
    binary_features = ['BASE_TYPE', 'COM_CAT#25', 'COM_CAT#26', "CSI"]
    numerical_features = set(X.columns) - set(categorical_features) - set(binary_features)

    X[categorical_features] = X[categorical_features].astype('int', errors='ignore')

    numerical_features = list(numerical_features)

    X[categorical_features] = X[categorical_features].astype('category', errors='ignore')

    X = X.drop(columns=profileReport.get_rejected_variables())
    categorical_features = list(set(X).intersection(categorical_features))
    numerical_features = list(set(X).intersection(numerical_features))
    binary_features = list(set(X).intersection(binary_features))

    classifier_pipeline = Pipeline(steps=[
        ('feature_processing', ColumnTransformer(transformers=[
            # binary
            ('binary', Pipeline([
                ('impute', SimpleImputer(missing_values=np.nan, strategy='most_frequent'))]),
             binary_features),

            # numeric
            ('numeric', Pipeline([
                ('impute', SimpleImputer(missing_values=np.nan, strategy='mean')),
                ('scale', RobustScaler()),
                ('transform', QuantileTransformer(output_distribution='normal')),
                ('engineer', PolynomialFeatures())]),
             numerical_features),

            # categorical
            ('categorical', Pipeline([
                ('impute', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-10000)),
                ('toint', FunctionTransformer(lambda x: x.astype('int64')))
            ]),
             categorical_features),

        ])),
    ]
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    X_train = classifier_pipeline.fit_transform(X_train)
    X_test = classifier_pipeline.fit_transform(X_test)
    X_submission_df, X_submission = process_submission_data(classifier_pipeline)
    return X_train, X_test, y_train, y_test, X_submission_df, X_submission


if __name__ == "__main__":
    process_data()