from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso, LassoCV
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

df = pd.read_excel('ModelExcel.xlsx', 'Desc')
df.drop(columns='Ligand_name', inplace=True)



#select subset of data
data_y = pd.DataFrame(df)
data_x = pd.DataFrame(df)


x = data_x.drop(columns='AP')
y = df["AP"]


# train test split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

#pipeline with scaler and LASSO
pipeline = Pipeline([('scaler', StandardScaler()), ('model', Lasso(selection='random', max_iter=5000))])

#crossvalidation method

cv = RepeatedKFold(n_splits=5, n_repeats=3)

#cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

#GridSearch to find alpha

search = GridSearchCV(pipeline,
                      {'model__alpha':np.arange(0.001,1,0.01)},
                      cv = cv, scoring="neg_mean_squared_error",verbose=3
                      )


search.fit(xtrain,ytrain)

coefficients = search.best_estimator_.named_steps['model'].coef_
intercept = search.best_estimator_.named_steps['model'].intercept_



importance = np.abs(coefficients)

#save output to txt
file_path ='Lasso2.txt'
sys.stdout = open(file_path, "w")

#score = model.score(xtrain, ytrain)
print(search.best_score_)
print(search.best_params_)
print(intercept)
print(coefficients)


print(list(zip(np.array(xtrain.columns)[importance>0], np.array(coefficients)[importance>0])))

#eli5.show_weights(search, top=-1, feature_names = xtrain.columns.tolist())

#plot predicted vs train and test

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
