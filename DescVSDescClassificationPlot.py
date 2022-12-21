import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.svm import SVR


#read in data
df = pd.read_excel('ModelExcel.xlsx', 'Desc')
#df.drop(columns='Ligand_name', inplace=True)

# read in screening data, plotted with red. Could be edited to not plot screening data...

df2 = pd.read_excel('First_Descriptors_ready_v2.xlsx', 'PScreen')

#Descriptors to plot

n = df['Ligand'].tolist()

x = df['Ni_VBur_Boltz']

y = df["Ni_electrophilicity_Boltz"]

#Descriptors to plot

n2 = df2['Ligand_name'].tolist()

x2 = df2['Ni_VBur_Boltz']

y2 = df2["Ni_electrophilicity_Boltz"]

#figure details

plt.figure(figsize=(20,20))
plt.scatter(x, y, s=5, color="blue", label="Ligand")
plt.scatter(x2, y2, s=5, color="red", label="Screening Ligand")

# dot name assignation 

for i, label in enumerate(n):
    plt.text(x[i], y[i], s=label)


for i, label in enumerate(n2):
    plt.text(x2[i], y2[i], s=label)
    
#labels

plt.xlabel("Ni_VBur_Boltz")
plt.ylabel("Ni_electrophilicity_Boltz")
#plt.plot(ytrain, y_train_pred, lw=0.8, color="red", label="predicted")
plt.legend()


plt.show()
