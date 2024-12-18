from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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