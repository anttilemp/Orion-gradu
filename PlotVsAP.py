from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.svm import SVR


#read in data
df = pd.read_excel('RxnScreenDesc.xlsx', 'Desc')
df.drop(columns='Ligand_name', inplace=True)

x = df[['Ni_charge_Boltz']]

y = df["AP"]


plt.scatter(x, y, s=5, color="blue", label="CC")

plt.xlabel("Descriptor")
plt.ylabel("AP")
#plt.plot(ytrain, y_train_pred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()