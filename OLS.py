import pandas as pd
import pingouin as pg
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

#Ordinary linear regression. Should be rewrited...

df = pd.read_excel('File.xlsx')

#choose descriptors

b = pd.DataFrame(df[['AP','x135', 'l10', 'd61', 'd13']])
x_columns = ['x135', 'l10', 'd61', 'd13']
y = b["AP"]

# function to get model statistics

def get_stats():
    x = b[x_columns]
    x = sm.add_constant(x)
    results = sm.OLS(y, x).fit()
    print(results.summary())
get_stats()
