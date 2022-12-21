import pandas as pd
import pingouin as pg
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = pd.read_excel('Model1Descriptors.xlsx')
df.drop(columns='Reaction', inplace=True)
df.drop(columns='Ligand', inplace=True)
df.drop(columns='Solvent', inplace=True)
#b = pd.DataFrame(df[['AP','x135', 'd13','d66','HBA','x26']])
b = pd.DataFrame(df[['AP','x135', 'l10', 'd61', 'd13']])

#x_columns = ['x135','d13','d66','HBA','x26']
x_columns = ['x135', 'l10', 'd61', 'd13']

y = b["AP"]

## creating function to get model statistics

def get_stats():
    x = b[x_columns]
    x = sm.add_constant(x)
    results = sm.OLS(y, x).fit()
    print(results.summary())
get_stats()