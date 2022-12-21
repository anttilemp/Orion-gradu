import pandas as pd
import pingouin as pg

df = pd.read_excel('ModelExcel.xlsx')

df.drop(columns='Ligand_name', inplace=True)


correlations = pg.pairwise_corr(df, columns='AP')

with pd.ExcelWriter('output.xlsx') as writer1:
    correlations.to_excel(writer1, startrow=0, startcol=0, index=False)

