from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess(df):
    
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Substituir valores ausentes (pd.NA) pela média da respectiva coluna
    df.fillna(df.mean(), inplace=True)
    
    # Transformar variáveis categóricas em colunas numéricas binárias
    X = pd.get_dummies(df)
        
    # Verifique se há valores ausentes após as transformações
    if X.isnull().values.any():
        print("Ainda existem valores NaN no DataFrame!")
        X = X.dropna()

    # Normalizar os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) 
    
    return X_scaled

def plot_pca(X_scaled, labels, file_name, title):
    # Reduzir os dados para 2D com PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(X_scaled)

    unique_labels = np.unique(labels)  # Identificar os clusters únicos
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))  # Definir cores

    plt.figure(figsize=(8, 6))

    # Plotar cada cluster separadamente para permitir a legenda
    for label, color in zip(unique_labels, colors):
        mask = (labels == label)  # Filtrar os pontos do cluster atual
        
        if label == -1:  # Outliers identificados pelo DBSCAN
            color = "red"  # Define cor vermelha para os outliers
            label_name = "Outliers"
        else:
            label_name = f"Cluster {label}"
        
        plt.scatter(reduced_data[mask, 0], reduced_data[mask, 1], 
                    color=color, label=label_name, alpha=0.7)
        
    plt.title(title)
    plt.legend()
    plt.savefig(file_name)
    plt.close()
    
    
def plot_graph_kmeans (X, labels, centroids, file_name, title):
        
    labels = np.array(labels) 
    labels = np.array(labels).astype(str)  # Converta para string para evitar problemas com Seaborn

    # Criar scatter plot dos clusters
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette="viridis", legend="full")

    # # Plotar os centróides
    # plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label="Centroids")

    plt.title(title)
    plt.legend()
    plt.savefig(file_name)
    plt.close()

def plot_dbscan_clusters(X, labels, file_name, title):
    """
    Plota os clusters gerados pelo DBSCAN.
    
    Parâmetros:
    - X: array numpy (n_samples, 2), dados originais em 2D.
    - labels: array numpy (n_samples,), rótulos dos clusters gerados pelo DBSCAN.
    """
    unique_labels = set(labels)  # Identificar clusters únicos e outliers
    palette = sns.color_palette("viridis", len(unique_labels))  # Paleta de cores

    plt.figure(figsize=(8, 6))

    for label, color in zip(unique_labels, palette):
        if label == -1:
            color = "red"  # Outliers em vermelho
            label_name = "Outliers"
        else:
            label_name = f"Cluster {label}"
        
        plt.scatter(X[labels == label, 0], X[labels == label, 1], 
                    c=[color], label=label_name, alpha=0.6, edgecolors="k")

    plt.title(title)    
    plt.legend()
    plt.savefig(file_name)