import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.cluster import DBSCAN

# Lista com os nomes das colunas
collumns = ['date', 'time', 'epoch', 'moteid', 'temperature', 'humidity', 'light', 'voltage']

# Carregar dados
df = pd.read_csv("../../data.txt", sep=" ", engine='python', names=collumns)
df_1000 = df.iloc[:50000].copy()

# Limpeza básica - substituindo "?" por NaN e removendo linhas faltantes
df_1000.replace("?", np.nan, inplace=True)
df_1000 = df_1000.dropna()

# Remove Coluna Target
df2 = df_1000.drop(columns=['date','time', 'epoch', 'moteid',])

# Transformar variáveis categóricas em colunas numéricas binárias
X = pd.get_dummies(df2)

# Certificar-se de que todos os dados estão em formato numérico
X = X.apply(pd.to_numeric)  

# Verifique se há valores ausentes após as transformações
if X.isnull().values.any():
    print("Ainda existem valores NaN no DataFrame!")
    X = X.dropna()

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Testando diferentes valores de eps
# for eps in [0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
#     dbscan = DBSCAN(eps=eps, min_samples=3)
#     clusters = dbscan.fit_predict(X_scaled)
#     print(f'Para eps={eps}, clusters formados:')
#     print(np.unique(clusters, return_counts=True))

for min_samples in [2, 3, 4, 5, 10, 15]:
    dbscan = DBSCAN(eps=0.5, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_scaled)
    print(f'Para min_samples={min_samples}, clusters formados:')
    print(np.unique(clusters, return_counts=True))