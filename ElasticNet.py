from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso, LassoCV, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.preprocessing import PolynomialFeatures



#read in data
df = pd.read_excel('RxnScreenDesc.xlsx', 'Desc')
df.drop(columns='Ligand_name', inplace=True)


#df2 = pd.read_excel('Descriptorsfromlasso.xlsx', 'Sheet6')

#select subset of data
data_y = pd.DataFrame(df)
data_x = pd.DataFrame(df)
#define predictor and response variables
#numerics = ['int16','int32','int64','float16','float32','float64']
#numerical_vars = list(data_x.select_dtypes(include=numerics).columns)
#data = data_x[numerical_vars]

#vals = df2['names6'].values.tolist()

#x = df[vals]

x = data_x.drop(columns='AP')

#x = pd.read_excel('RemoveCorrelated3.xlsx')

y = df["AP"]



xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=2)

pipeline = Pipeline([('scaler', StandardScaler()), ('model', ElasticNet(max_iter=5000))])

#cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=123)

cv = RepeatedKFold(n_splits=5, n_repeats=3)

search = GridSearchCV(pipeline,
                      {'model__alpha': np.arange(1,10,1), 'model__l1_ratio': np.arange(0.001,1,0.1)},
                      cv = cv, scoring="neg_root_mean_squared_error",verbose=3
                      )


search.fit(xtrain,ytrain)

coefficients = search.best_estimator_.named_steps['model'].coef_
intercept = search.best_estimator_.named_steps['model'].intercept_



importance = np.abs(coefficients)


file_path ='ElasticNet.txt'
#file_path ='Run8_fromLassoKarsinta2.txt'
sys.stdout = open(file_path, "w")

#score = model.score(xtrain, ytrain)
print(search.best_score_)
print(search.best_params_)
print(intercept)
print(coefficients)


print(list(zip(np.array(xtrain.columns)[importance>0], np.array(coefficients)[importance>0])))

#eli5.show_weights(search, top=-1, feature_names = xtrain.columns.tolist())

y_train_pred = search.predict(xtrain)
r1 = r2_score(ytrain, y_train_pred)
print('Train set R2',r1)
print('Train set:', ytrain)

y_test_pred = search.predict(xtest)
r1 = r2_score(ytest, y_test_pred)
print('Test set R2', r1)
print('Test set:', ytest)
print('Test set pred:', y_test_pred)


plt.scatter(ytrain, y_train_pred, s=5, color="blue", label="Train")
plt.scatter(ytest, y_test_pred, s=5, color="red", label="Test")
plt.xlabel("Measured AP")
plt.ylabel("Predicted AP")
#plt.plot(ytrain, y_train_pred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()