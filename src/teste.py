import pandas as pd
import numpy as np

# Simulação do seu DataFrame com <NA> como valores nulos
data = {
    0: [68, 68, 66, 68, 68, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
    1: [97, 96, 97, 97, 99, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
    2: [17.803, 18.7928, 20.939, 23.6144, 19.4004, 18.391, 22.5756, 21.6054, 23.3694, 23.2714]
}

df = pd.DataFrame(data)
print (df)
# Substituindo os valores <NA> pela média da respectiva coluna
df.fillna(df.mean(), inplace=True)

# Exibindo o DataFrame resultante
print ("após")
print(df)