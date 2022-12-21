import pandas as pd
import pingouin as pg
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#read in data
df = pd.read_excel('ModelExcel.xlsx', 'Desc')
df.drop(columns='Ligand_name', inplace=True)

#df2 = pd.read_excel('Descriptorsfromlasso.xlsx')

#vals = df2['names'].values.tolist()

#final = df[vals].corr()

data_x = pd.DataFrame(df)

numerics = ['int16','int32','int64','float16','float32','float64']
numerical_vars = list(data_x.select_dtypes(include=numerics).columns)

df2 = df[numerical_vars]

#pairwise correlations

correlations = pg.pairwise_corr(df2, columns='AP')

#select top 75 or desired amount

best = correlations.nlargest(75, "power")
a = best['Y']
matrix = a.values.tolist()

final = df[matrix].corr()

# you should change these to edit image details

plt.figure(figsize=(100, 80))
cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="light", as_cmap=True)
mask = np.triu(np.ones_like(final, dtype=bool))
fig, ax = plt.subplots(figsize=(100,80))
plot = sns.heatmap(final, mask=mask, center=0, annot=True, fmt='.2f', square=True, cmap=cmap, ax=ax)
fig.set_size_inches(100, 80)
plt.savefig("Top75.png",bbox_inches='tight',dpi=300)
