import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from kneed import KneeLocator

def preprocess(df):
    
    # Substituir valores ausentes (pd.NA) pela média da respectiva coluna
    df = df.fillna(df.mean())
    
    # Transformar variáveis categóricas em colunas numéricas binárias
    X = pd.get_dummies(df)
    
    # Certificar-se de que todos os dados estão em formato numérico
    X = X.apply(pd.to_numeric)  

    # Verifique se há valores ausentes após as transformações
    if X.isnull().values.any():
        print("Ainda existem valores NaN no DataFrame!")
        X = X.dropna()

    # Normalizar os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled


def elbow(X_scaled):
       
    inertias = []
    wcss = []
    range_clusters = range(1, 11)

    for k in range_clusters:
        kmeans = KMeans(n_init='auto', n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        wcss.append(kmeans.inertia_)
           
    # Usar o kneed para encontrar o cotovelo
    knee_locator = KneeLocator(range_clusters, wcss, curve="convex", direction="decreasing")
    optimal_k = knee_locator.knee

    # Exibir o número ideal de clusters
    # print(f"Número ideal de clusters: {optimal_k}")
    return optimal_k

    # # Plotar o método do cotovelo
    # plt.plot(range_clusters, inertias, 'o--')
    # plt.xlabel('Número de Clusters')
    # plt.ylabel('Inércia')
    # plt.title('Método do Cotovelo para Número Ideal de Clusters')
    # plt.savefig('plot_inertias.png')
    # plt.close()

    # pd.Series(inertias).diff().plot(kind='bar')
    # plt.savefig('bar_inertias.png')


def run_kmeans(X_scaled, k):
            
    # n_clusters=5: Define o número de clusters que o algoritmo tentará encontrar.
    # random_state=42: Garante reprodutibilidade ao inicializar os centroids com a mesma semente pseudoaleatória.
    # n_init='auto': No Scikit-learn 1.4 ou mais recente, o valor 'auto' define o número de inicializações 
    # automaticamente (geralmente otimizado para um número adequado). Antes dessa versão, o padrão era 10 inicializações.    
    kmeans = KMeans(n_init='auto', n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)

    # Verificar as labels geradas pelo K-Means
    # print("Labels de K-Means:", kmeans.labels_)

    silhouette = silhouette_score(X_scaled, kmeans.labels_)
    # print("Índice de Silhueta:", silhouette_avg)

    davies_bouldin = davies_bouldin_score(X_scaled, kmeans.labels_)
    # print("Índice de Davies-Bouldin:", davies_bouldin)

    calinski_harabasz = calinski_harabasz_score(X_scaled, kmeans.labels_)
    # print("Coeficiente de Calinski-Harabasz:", calinski_harabasz)

    params = (kmeans.labels_)
    results = (silhouette, davies_bouldin, calinski_harabasz)
    
    return (results, params)
    
    # pca(X_scaled, kmeans)

def pca(X_scaled, kmeans):
        # Reduzir os dados para 2D com PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(X_scaled)

    # Plotar os clusters
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans.labels_)
    plt.title("Visualização dos Clusters usando PCA")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    # plt.show()
    plt.savefig('plot_pca.png')
    plt.close()