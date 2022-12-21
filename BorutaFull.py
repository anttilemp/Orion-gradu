from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso, LassoCV, MultiTaskLasso, LassoLarsIC
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
from xgboost import XGBRegressor


#read in data
df = pd.read_excel('RxnScreenDesc.xlsx')

#select subset of data
data_y = pd.DataFrame(df)
data_x = pd.DataFrame(df)
#define predictor and response variables
numerics = ['int16','int32','int64','float16','float32','float64']
numerical_vars = list(data_x.select_dtypes(include=numerics).columns)
data = data_x[numerical_vars]

x = data.drop(columns='AP')

y = df["AP"]

#importance score method, default XGBregressor

rf = XGBRegressor(n_jobs=3, max_depth=1)

# define Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)


pipeline = Pipeline([('scaler', StandardScaler()), ('model', feat_selector)])
#('scaler', StandardScaler()),
pipeline.fit(x, y)


print(pipeline.named_steps['model'].support_)

# check ranking of features
print(pipeline.named_steps['model'].ranking_)

# call transform() on X to filter it down to selected features
X_filtered = pipeline.transform(x)
a = pd.DataFrame(X_filtered)


# zip my names, ranks, and decisions in a single iterable
feature_ranks = list(zip(x.columns,
                         pipeline.named_steps['model'].ranking_,
                         pipeline.named_steps['model'].support_))

# iterate through and print out the results
for feat in feature_ranks:
    print('Feature: {:<25} Rank: {},  Keep: {}'.format(feat[0], feat[1], feat[2]))

accept = x.columns[pipeline.named_steps['model'].support_].to_list()

print(accept)
"""
with pd.ExcelWriter('Boruta1.xlsx') as writer1:
    a.to_excel(writer1, sheet_name='Sheet1')
    accept.to_excel(writer1, sheet_name='Sheet2')
"""
