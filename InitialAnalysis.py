
# coding: utf-8

# In[158]:


import dask.dataframe as dd
import os
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
from sklearn.linear_model import LinearRegression, LogisticRegression

from hyperopt import hp, tpe
from hyperopt.fmin import fmin

from pandas_profiling import ProfileReport

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer, FunctionTransformer

from sklearn.model_selection import train_test_split

import lightgbm

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold, TimeSeriesSplit, cross_val_score

from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

from sklearn.metrics import roc_auc_score

from feature_selector import FeatureSelector

from catboost import CatBoostClassifier

from sklearn.linear_model import ElasticNet
from sklearn.svm import SVC

from catboost import CatBoostClassifier, Pool, cv
from sklearn.metrics import accuracy_score

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


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


# In[3]:


def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True, sep=';')
    df = reduce_mem_usage(df)
    return df


# In[2]:


def sortByDate(df, validateSet=False):
    df['SNAP_DATE'] = pd.to_datetime(df['SNAP_DATE'], format='%d.%m.%y')
    df['BASE_TYPE'] -= 1
    if validateSet:
        df = df.groupby(by="SK_ID", group_keys=False).apply(lambda grp: grp.nlargest(1, 'SNAP_DATE'))
    return df#.groupby(by="SK_ID", group_keys=False).apply(lambda grp: grp.nlargest(1, 'SNAP_DATE'))


# In[5]:


df = import_data('condition_csi/dataset/bs_avg_kpi.csv')


# In[61]:


subs_csi = pd.read_csv('train/subs_csi_train.csv', index_col='SK_ID')

subs_features = pd.read_csv('train/subs_features_train.csv', index_col='SK_ID')

subs_bs_consumption = pd.read_csv('train/subs_bs_consumption_train.csv')
#subs_bs_data_session_train = pd.read_csv('train/subs_bs_data_session_train.csv', index_col='SK_ID')
#subs_bs_voice_session_train = pd.read_csv('train/subs_bs_voice_session_train.csv', index_col='SK_ID')


# In[62]:


df =subs_csi.merge(subs_features, on="SK_ID")


# In[63]:


df = df.merge(subs_bs_consumption.groupby(by=["SK_ID", "MON"], as_index=False).sum().set_index('SK_ID'), on='SK_ID')


# In[64]:


df = sortByDate(df)


# In[65]:


df.head()


# In[66]:


df=df[df['ACT']==1]


# In[67]:


X, y = df.drop(columns=["CSI", "SNAP_DATE", "CONTACT_DATE", 'COM_CAT#24', 'ACT', 'MON', 'CELL_LAC_ID']), df["CSI"]


# In[68]:


X.shape


# In[11]:


profileReport = ProfileReport(df)
profileReport


# In[69]:


categorical_features = ['ARPU_GROUP','DEVICE_TYPE_ID', 'INTERNET_TYPE_ID'] #['ARPU_GROUP', 'COM_CAT#1', 'COM_CAT#2', 'COM_CAT#3', 'COM_CAT#7', 'COM_CAT#34', 'DEVICE_TYPE_ID', 'INTERNET_TYPE_ID']
binary_features = ['BASE_TYPE', 'COM_CAT#25', 'COM_CAT#26', "CSI"]
numerical_features = set(X.columns) - set(categorical_features) - set(binary_features)


# In[86]:


X[categorical_features] = X[categorical_features].astype('int', errors='ignore')

numerical_features = list(numerical_features)

X[categorical_features] = X[categorical_features].astype('category', errors='ignore')


# In[71]:


profileReport.description_set['variables']['type'] == 'BOOL'


# In[358]:


fs = FeatureSelector(data = X, labels = y)

fs.identify_all(selection_params = {'missing_threshold': 0.6, 'correlation_threshold': 0.98, 
                                    'task': 'classification', 'eval_metric': 'auc', 
                                     'cumulative_importance': 0.99})
fs.plot_feature_importances(threshold = 0.99, plot_n = 12)


# In[359]:


train_removed_all_once = fs.remove(methods = 'all', keep_one_hot = False)


# In[13]:


len(X.columns) - len(train_removed_all_once.columns)


# In[72]:


X = train_removed_all_once


# In[73]:


profileReport.get_rejected_variables()


# In[87]:


X = X.drop(columns=['COM_CAT#22','COM_CAT#23', 'COM_CAT#28'])


# In[88]:


categorical_features = list(set(X).intersection(categorical_features))
numerical_features = list(set(X).intersection(numerical_features))
binary_features = list(set(X).intersection(binary_features))
categorical_indices = np.where(X.dtypes == 'category')


# In[89]:


X[categorical_features].head()


# In[90]:


X.shape


# In[ ]:


len(categorical_indices[0]) == len(categorical_features)


# In[91]:


classifier_pipeline = Pipeline(steps = [       
    ('feature_processing', ColumnTransformer(transformers = [        
            #binary
            ('binary', Pipeline([
                ('impute', SimpleImputer(missing_values=np.nan, strategy='most_frequent'))]),
            binary_features), 
                    
            #numeric
            ('numeric', Pipeline([
                ('impute', SimpleImputer(missing_values=np.nan, strategy='mean')),
                ('scale', RobustScaler()),
                ('transform', QuantileTransformer(output_distribution='normal')),
                ('engineer', PolynomialFeatures())]),
            numerical_features),
        
            #categorical
            ('categorical', Pipeline([
                ('impute', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-10000)),
                ('toint', FunctionTransformer(lambda x: x.astype('int64')))
]),
             #                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
            categorical_features),
            
    ])),
    ]
)

CatBoostClassifier()


# In[92]:


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)


# In[93]:


X_train = classifier_pipeline.fit_transform(X_train)
X_test = classifier_pipeline.fit_transform(X_test)


# In[94]:


len(X_train[0,:])


# In[95]:


catboost_classifier = CatBoostClassifier(iterations=4000, random_seed=666, 
                                        depth=4, learning_rate=0.2,
                                        use_best_model=True,
                                        best_model_min_trees=40,
                                        od_type="Iter",
                                        l2_leaf_reg=7,
                                        od_wait=500,
                                        loss_function='Logloss', eval_metric='AUC')


# In[96]:


catboost_classifier.fit(X=X_train, y=y_train, plot=True, verbose=False, eval_set=(X_test, y_test), cat_features=np.arange(381, 384))


# In[100]:


#train_data = lightgbm.Dataset(X_train, label=y_train)
#test_data = lightgbm.Dataset(X_test, label=y_test, reference=train_data)


# In[121]:


parameters = {
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 31,
    'num_estimators': 20,
    'verbose': 0
}


# In[161]:


model = lightgbm.LGBMClassifier(objective='binary', random_state=888, n_estimators=500,
        learning_rate=0.01, num_leaves=124, colsample_bytree=0.996)


# In[162]:


model.fit(X_train, y_train,
                       eval_set=[(X_test, y_test)],
                       eval_metric='AUC')


# In[160]:


def objective(params):
    params = {
        'num_leaves': int(params['num_leaves']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
    }
    
    clf = lightgbm.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.01,
        **params
    )
    
    score = cross_val_score(clf, X_train, y_train,  scoring='roc_auc',cv=StratifiedKFold()).mean()
    print("ROC AUC {:.3f} params {}".format(score, params))
    return score

space = {
    'num_leaves': hp.quniform('num_leaves', 8, 128, 2),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
}

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=10)


# In[413]:


pickle_object = np.concatenate((X_test, np.array(y_test.tolist()).reshape(34889, 1)), axis=1)


# In[414]:


pickle_object.shape


# In[427]:


import pickle
with open("validateDF.pkl", 'wb+') as f:
    pickle.dump(X_validate, f)


# In[247]:


catboost_classifier.fit(X_train, y_train, eval_set = (X_test, y_test), verbose=False, plot=True)


# In[30]:


svc_class = SVC(random_state=444, C=20, verbose=True, probability=True)
svc_class.fit(X_train, y_train)


# In[22]:


tuned_parameters = [
    {
#         'feature_processing__numeric__impute': [
#             SimpleImputer(missing_values=np.nan, strategy='mean'),
#             SimpleImputer(missing_values=np.nan, strategy='median'),
#             SimpleImputer(missing_values=np.nan, strategy='most_frequent')
#         ],
#         'feature_processing__numeric__scale': [
#             MinMaxScaler(),
#             MaxAbsScaler(),
#             RobustScaler(),
#             StandardScaler()
#         ],
#         'feature_processing__numeric__transform': [
#             QuantileTransformer(output_distribution='normal'),
#             PowerTransformer()
#         ], 
         'C': [0.05, 0.1, 0.5, 1, 10, 40],
         'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
#         'solver': ['newton-cg', 'lbfgs', 'sag']
    }
]
grid = GridSearchCV(SVC(random_state=954, class_weight='balanced'), tuned_parameters,
                   scoring='roc_auc', cv=5, verbose=10, n_jobs=-1)


# In[ ]:


grid.fit(X_train, y_train)


# In[58]:


grid.best_params_


# In[ ]:


res[:,1]


# In[163]:


res = model.predict_proba(X_train)
print("ROC AUC for train")
roc_auc_score(y_train, res[:,1])


# In[164]:


res = model.predict_proba(X_test)
print("ROC AUC for test")
roc_auc_score(y_test, res[:,1])


# In[166]:


subs_csi_1 = pd.read_csv('test/subs_csi_test.csv', index_col='SK_ID')

subs_features_1 = pd.read_csv('test/subs_features_test.csv', index_col='SK_ID')
subs_bs_consumption1 = pd.read_csv('test/subs_bs_consumption_test.csv')

df1 =subs_csi_1.merge(subs_features_1, on="SK_ID")


# In[167]:


df1 = df1.merge(subs_bs_consumption1.groupby(by=["SK_ID", "MON"], as_index=False).sum().set_index('SK_ID'), how='left', on='SK_ID')


# In[168]:


df1.shape


# In[132]:


df1 = df1.fillna(method='backfill')


# In[169]:


df1 = sortByDate(df1)


# In[170]:


X_validate= df1.drop(columns=["SNAP_DATE", "CONTACT_DATE", 'COM_CAT#24'])


# In[171]:


X_validate.columns


# In[172]:


X_validate = classifier_pipeline.fit_transform(X_validate)


# In[173]:


X_validate.shape


# In[174]:


res = model.predict_proba(X_validate)


# In[175]:


res.shape


# In[176]:


res = res[:,1]


# In[177]:


res


# In[178]:


X_validate = df1.drop(columns=["SNAP_DATE", "CONTACT_DATE", 'COM_CAT#24'])


# In[430]:


X_validate.to_pickle("./validDF1.pkl")


# In[179]:


X_validate["pred"] = res

X_validate = X_validate["pred"].groupby(by="SK_ID").agg({'pred': np.mean})

X_validate = X_validate.values


# In[180]:


len(X_validate.T[0])


# In[181]:


np.savetxt("foo.csv", X_validate.T[0], delimiter="\n")


# In[ ]:


def transform_x(x):
    if type(x) == float and np.isnan(x):
        return x
    return float(x.replace(',', '.'))


# In[ ]:


train_data=os.listdir('condition_csi/dataset/train')
test_data=os.listdir('condition_csi/dataset/test')


# In[ ]:


train_data


# In[ ]:


for dfn in tqdm(test_data):
    df = pd.read_csv('condition_csi/dataset/test/' + dfn, sep=";")
    for i, j in tqdm(zip(df.dtypes, df.columns)):
        if i == 'object' and "DATE" not in j and "TIME" not in j and "ID" not in j:
            df[j] = df[j].apply(transform_x)
    df.to_csv(dfn, index=False)


# # Test

# In[ ]:


import hashlib
import math

def F(n):
    if n == 0: return 0
    elif n == 1: return 1
    else: return F(n-1)+F(n-2)


password = input().strip()

if password[-1].isdigit():
    digit = int(password[-1])
    password += str(F(digit))
else:
    password = password[:-1] + password[-1].swapcase()

print(hashlib.sha256(password.encode('utf-8')).hexdigest())


# In[ ]:


from socket import inet_ntoa
from struct import pack

def calcDottedNetmask(mask):
    bits = 0xffffffff ^ (1 << 32 - mask) - 1
    return inet_ntoa(pack('>I', bits))

address = input()
mask = int(input())
subnet_mask = calcDottedNetmask(mask)
ip = address.strip().split('.')
mask = subnet_mask.strip().split('.')

ans = []
for i, m in zip(ip, mask):
    ans.append(int(i) & int(m))

print(*ans, sep='.')


# In[ ]:


from statistics import mean
from decimal import Decimal, ROUND_HALF_UP

temperatures = list(map(int, input().split(', ')))
mean_number = Decimal(mean(temperatures))
print(mean_number.quantize(Decimal('1'), rounding=ROUND_HALF_UP), end=' ')

from collections import Counter
def perseus_sort(l):
    counter = Counter(l)
    counter = [(v, k) for k, v in counter.items()]
    return sorted(counter, key=lambda x: (x[0], -x[1]))


res = perseus_sort(temperatures)
ans = []
for freq in res:
    ans.append(freq[1])

print(*ans, sep=' ')

