import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

# def eps(X_scaled):   

#     for eps in [0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
#         dbscan = DBSCAN(eps=eps, min_samples=3)
#         clusters = dbscan.fit_predict(X_scaled)
#         print(f'Para eps={eps}, clusters formados:')
#         print(np.unique(clusters, return_counts=True))

# def min_samples(X_scaled):
#     for min_samples in [2, 3, 4, 5, 10, 15]:
#         dbscan = DBSCAN(eps=0.5, min_samples=min_samples)
#         clusters = dbscan.fit_predict(X_scaled)
#         print(f'Para min_samples={min_samples}, clusters formados:')
#         print(np.unique(clusters, return_counts=True))
        
# Função para encontrar os melhores parâmetros
def find_best_params(X, eps_values, min_samples_values):
    best_params = None
    best_score = -1
    # best_score = float('inf')
    # best_labels = None
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
                    
            if len(set(labels)) > 1:
                silhouette = silhouette_score(X, labels)                
                if silhouette > best_score:                    
                    best_score = silhouette
                    best_params = (eps, min_samples)                                        
    return (best_params)     
    

def estimate_min_samples(X, neighbors_range=[4, 5, 10, 16, 20]):
    best_min_samples = None
    best_variability = float('inf')  # Inicializa com um valor alto

    # Define the range of neighbors
    neighbors_range = range (4, 21)
    
    for n_neighbors in neighbors_range:
        
        # Computing nearest neighbors
        neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
        distances, _ = neighbors.kneighbors(X)

        # Computing the average density and estimate the min_samples
        avg_density = np.mean(distances[:, -1])  # Média da distância ao k-ésimo vizinho
        estimated_min_samples = int(np.clip(avg_density * X.shape[1], 2, 20))  # Limita entre 2 e 20
        
        # Compute the variability by standard deviation 
        variability = np.std(distances[:, -1])  # Desvio padrão das distâncias
        
        # Choosing the best min_samples
        if variability < best_variability:
            best_variability = variability
            best_min_samples = estimated_min_samples

    return best_min_samples


def find_optimal_epsilon(X, min_samples):
    
    # Compute nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(X)
    
    # Compute the distances to the k-ésimo nearest neighbors
    distances, _ = neighbors.kneighbors(X)
    distances = np.sort(distances[:, -1])  # Ordenar as distâncias do k-vizinho
    
    # Find the knee point
    knee = KneeLocator(range(len(distances)), distances, curve="convex", direction="increasing")
    
    return distances[knee.knee]  # Retorna o melhor epsilon estimado


def run (X, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    unique, counts = np.unique(labels, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    
    # Contar o número de outliers
    n_outliers = np.sum(labels == -1)
    if len(set(labels)) > 1:
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
        results = (silhouette, davies_bouldin, calinski_harabasz)
        params = (n_clusters, n_outliers,cluster_counts, labels)
        return (results, params)
    else:     
        return None
