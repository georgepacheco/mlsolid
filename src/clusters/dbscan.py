import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import ParameterGrid


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

def eps(X_scaled):   

    for eps in [0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        dbscan = DBSCAN(eps=eps, min_samples=3)
        clusters = dbscan.fit_predict(X_scaled)
        print(f'Para eps={eps}, clusters formados:')
        print(np.unique(clusters, return_counts=True))

def min_samples(X_scaled):
    for min_samples in [2, 3, 4, 5, 10, 15]:
        dbscan = DBSCAN(eps=0.5, min_samples=min_samples)
        clusters = dbscan.fit_predict(X_scaled)
        print(f'Para min_samples={min_samples}, clusters formados:')
        print(np.unique(clusters, return_counts=True))
        
# Função para encontrar os melhores parâmetros
def find_best_params(X, eps_values, min_samples_values):
    best_params = None
    best_score = -1

    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            # Contar o número de outliers
            n_outliers = np.sum(labels == -1)
            
            # Ignorar casos onde todos os pontos estão no mesmo cluster ou são outliers
            if len(set(labels)) > 1:
                score = silhouette_score(X, labels)
                davies_bouldin = davies_bouldin_score(X, labels)
                calinski_harabasz = calinski_harabasz_score(X, labels)
                if score > best_score:                    
                    best_results = (score, davies_bouldin, calinski_harabasz)
                    best_params = (eps, min_samples, n_clusters, n_outliers)                    

    return (best_params, best_results)

def find_auto_params (X, eps_values, min_samples_values):
    # Grade de parâmetros
    param_grid = {'eps': eps_values, 'min_samples': min_samples_values}

    # Busca exaustiva
    best_params = None    
    best_score = -1
    for params in ParameterGrid(param_grid):
        dbscan = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
        labels = dbscan.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # Contar o número de outliers
        n_outliers = np.sum(labels == -1)
            
        if len(set(labels)) > 1:  # Garantir que há mais de um cluster
            score = silhouette_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)
            calinski_harabasz = calinski_harabasz_score(X, labels)
            if score > best_score:
                best_results = (score, davies_bouldin, calinski_harabasz)                
                best_params = (params, n_clusters, n_outliers)                
                
    # davies_bouldin = davies_bouldin_score(X, best_lables)
    # calinski_harabasz = calinski_harabasz_score(X, best_lables)
    # outlier_ratio = np.sum(best_lables == -1) / len(best_lables)

    return (best_params, best_results)
    # print("Melhores parâmetros:", best_params)
    # print("Melhor Silhouette Score:", best_score)
    # print ("Davies Bouldin: ", davies_bouldin)
    # print ("Calinski Harabasz: ", calinski_harabasz)
    # print ("Outilier ratio: ", outlier_ratio)
