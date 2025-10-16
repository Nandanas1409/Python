import pandas as pd
import numpy as np

data = {'category' : ['a','b','c','a'],
        'value' : [45,22,64,28]}
df = pd.DataFrame(data)

nominal_cols = ['category']
numeric_cols = ['value']

for col in numeric_cols :
    max = df[col].max()
    min = df[col].min()
    df[col] = (df[col]-min)/(max-min)

n = len(df)
num_sim = np.zeros((n,n))
nom_sim = np.zeros((n,n))
num_dissim = np.zeros((n,n))
nom_dissim = np.zeros((n,n))
mixed_sim = np.zeros((n,n))
mixed_dissim = np.zeros((n,n))

for i in range(n):
    for j in range(n):
        nom_match = 0
        for col in nominal_cols:
            if df.loc[i,col] == df.loc[i,col]:
                nom_match += 1
        nom_sim[i][j] = nom_match/len(nominal_cols)
        nom_dissim[i][j] = 1-nom_sim[i][j]

        num_sum = 0
        for col in numeric_cols:
            diff = abs(df.loc[i,col]-df.loc[j,col])
            num_sum += (1-diff)

        num_sim[i][j] = num_sum/len(numeric_cols)
        num_dissim[i][j] = 1-num_sim[i][j]

        mixed_sim[i][j] = (num_sim[i][j]+nom_sim[i][j])/2
        mixed_dissim[i][j] = 1-mixed_sim[i][j]

print("Numeric similarity : \n")
print(num_sim)
print("Numeric dissimilarity : \n")
print(num_dissim)
print("Nominal similarity : \n")
print(nom_sim)
print("Nominal dissimilarity : \n")
print(nom_dissim)
print("Mixed similarity : \n")
print(mixed_sim)
print("Mixed dissimilarity : \n")
print(mixed_dissim)