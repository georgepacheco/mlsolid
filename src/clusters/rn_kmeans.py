import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score


# # Carregar os dados de exemplo
# df = pd.read_csv("../heart.csv")
# df2=df.drop(columns=['DEATH_EVENT'])
# X = pd.get_dummies(df);
# scaled_X = StandardScaler().fit_transform(X)  # Normalização dos dados

# Lista com os nomes das colunas
collumns = ['date', 'time', 'epoch', 'moteid', 'temperature', 'humidity', 'light', 'voltage']

# Carregar dados
df = pd.read_csv("../data.txt", sep=" ", engine='python', names=collumns)
df_1000 = df.iloc[:100000]

# Limpeza básica - substituindo "?" por NaN e removendo linhas faltantes
df_1000.replace("?", np.nan, inplace=True)
df_1000.dropna(inplace=True)

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

# Definir um modelo simples de `tf.keras`
model = tf.keras.models.Sequential([
    layers.Dense(32, activation='relu', input_shape=(4,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu')  # Exemplo de 10 classes, ajuste conforme necessário
])

# Extrair features usando o modelo
feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
features = feature_extractor.predict(X_scaled)  # Obter as features

# inertias = []
# range_clusters = range(2, 11)

# for k in range_clusters:
#     # Aplicar o K-Means nas features extraídas
#     kmeans = KMeans(n_init='auto', n_clusters=k, random_state=42)
#     kmeans.fit(features)
#     inertias.append(kmeans.inertia_)

# print ("KMeans Inertia: ", inertias)

# # Plotar o método do cotovelo
# plt.plot(range_clusters, inertias, 'o--')
# plt.xlabel('Número de Clusters')
# plt.ylabel('Inércia')
# plt.title('Método do Cotovelo para Número Ideal de Clusters')
# plt.savefig('plot_inertias_rn.png')
# plt.close()

# pd.Series(inertias).diff().plot(kind='bar')
# plt.savefig('bar_inertias_rn.png')

kmeans = KMeans(n_init='auto', n_clusters=4, random_state=42)
kmeans.fit(features)
    
# Verificar as labels geradas pelo K-Means
print("Labels de K-Means:", kmeans.labels_)

silhouette_avg = silhouette_score(features, kmeans.labels_)
print("Índice de Silhueta:", silhouette_avg)

davies_bouldin = davies_bouldin_score(features, kmeans.labels_)
print("Índice de Davies-Bouldin:", davies_bouldin)

calinski_harabasz = calinski_harabasz_score(features, kmeans.labels_)
print("Coeficiente de Calinski-Harabasz:", calinski_harabasz)