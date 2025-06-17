
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict, field
from typing import List, Optional
from datetime import datetime
import json
import os

# Funções de normalização
def normalizeSilhouette(value):
    return (value + 1) / 2

def normalize_dbi(value):
    return 1 / (1 + value)

def normalize_chi(value, max_value):
    return value / max_value if max_value != 0 else 0

def choose_best_algorithm(silhouette_kmeans, silhouette_dbscan, dbi_kmeans, dbi_dbscan, chi_kmeans, chi_dbscan):
    sil_kmeans_norm = normalizeSilhouette(silhouette_kmeans)
    sil_dbscan_norm = normalizeSilhouette(silhouette_dbscan)

    dbi_kmeans_norm = normalize_dbi(dbi_kmeans)
    dbi_dbscan_norm = normalize_dbi(dbi_dbscan)

    max_chi = max(chi_kmeans, chi_dbscan)
    chi_kmeans_norm = normalize_chi(chi_kmeans, max_chi)
    chi_dbscan_norm = normalize_chi(chi_dbscan, max_chi)

    score_kmeans = (sil_kmeans_norm + dbi_kmeans_norm + chi_kmeans_norm) / 3
    score_dbscan = (sil_dbscan_norm + dbi_dbscan_norm + chi_dbscan_norm) / 3

    if score_kmeans > score_dbscan:
        best_algorithm = "K-Means"
        best_metric = sil_kmeans_norm
    else:
        best_algorithm = "DBSCAN"
        best_metric = sil_dbscan_norm

    best_scores = (score_kmeans, score_dbscan)
    metrics = (sil_kmeans_norm, sil_dbscan_norm)

    return best_algorithm, best_scores, metrics, best_metric

@dataclass
class Algorithms:
    name: Optional[str] = None
    clusters: Optional[int] = None
    silhouette_score: Optional[float] = None
    davies_bouldin: Optional[float] = None
    calisnky_harabasz: Optional[float] = None
    score: Optional[float] = None
    metric_value: Optional[float] = None
    clusters_count: Optional[int] = None
    eps: Optional[float] = None
    samples: Optional[int] = None
    outliers: Optional[int] = None

@dataclass
class Statistics:
    date: Optional[int] = None
    algorithms: List[Algorithms] = field(default_factory=list)
    best_algorithm: Optional[str] = None

# Carregar dataset
df = pd.read_csv('meu_dataset.csv')
features_to_scale = ['HumiditySensor', 'AirThermometer', 'CO_Sensor', 'LightSensor']
X = df.copy()
X[features_to_scale] = X[features_to_scale].fillna(X[features_to_scale].median())
scaler = StandardScaler()
X[features_to_scale] = scaler.fit_transform(X[features_to_scale])

# KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
X['KMeans_Cluster'] = kmeans.fit_predict(X[features_to_scale])
kmeans_sil = silhouette_score(X[features_to_scale], X['KMeans_Cluster'])
kmeans_db = davies_bouldin_score(X[features_to_scale], X['KMeans_Cluster'])
kmeans_ch = calinski_harabasz_score(X[features_to_scale], X['KMeans_Cluster'])

# DBSCAN
dbscan = DBSCAN(eps=0.7, min_samples=5)
X['DBSCAN_Cluster'] = dbscan.fit_predict(X[features_to_scale])
if len(set(X['DBSCAN_Cluster'])) > 1 and -1 in set(X['DBSCAN_Cluster']):
    dbscan_sil = silhouette_score(X[features_to_scale], X['DBSCAN_Cluster'])
    dbscan_db = davies_bouldin_score(X[features_to_scale], X['DBSCAN_Cluster'])
    dbscan_ch = calinski_harabasz_score(X[features_to_scale], X['DBSCAN_Cluster'])
else:
    dbscan_sil = dbscan_db = dbscan_ch = 0.0

best_algorithm, best_scores, metrics, best_metric = choose_best_algorithm(
    kmeans_sil, dbscan_sil, kmeans_db, dbscan_db, kmeans_ch, dbscan_ch
)

# Salvar Benchmark
timestamp = datetime.now().strftime('%Y%m%d')
os.makedirs(f"benchmark/{timestamp}", exist_ok=True)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X[features_to_scale])
X['PCA1'], X['PCA2'] = X_pca[:, 0], X_pca[:, 1]

# Gráficos
plt.figure(figsize=(8, 6))
plt.scatter(X['PCA1'], X['PCA2'], c=X['KMeans_Cluster'], cmap='tab10', s=30)
plt.title('K-Means Clustering (PCA Projection)')
plt.savefig(f'benchmark/{timestamp}/kmeans_clusters.png')

plt.figure(figsize=(8, 6))
plt.scatter(X['PCA1'], X['PCA2'], c=X['DBSCAN_Cluster'], cmap='tab10', s=30)
plt.title('DBSCAN Clustering (PCA Projection)')
plt.savefig(f'benchmark/{timestamp}/dbscan_clusters.png')

# Perfil KMeans
df['KMeans_Cluster'] = X['KMeans_Cluster']
profile = df.groupby('KMeans_Cluster')[['HumiditySensor', 'AirThermometer', 'CO_Sensor', 'LightSensor', 'OccupancyDetector']].agg(['mean', 'std', 'min', 'max', 'count'])
profile.to_csv(f'benchmark/{timestamp}/kmeans_cluster_profile.csv')


# Criar diretórios para benchmark e subpasta por data

dbscan_labels = X['DBSCAN_Cluster']
dbscan_n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
dbscan_outliers = list(dbscan_labels).count(-1)

kmeans_data = Algorithms(
    name="KMeans",
    clusters=3,
    silhouette_score=round(kmeans_sil, 4),
    davies_bouldin=round(kmeans_db, 4),
    calisnky_harabasz=round(kmeans_ch, 4),
    score=round(best_scores[0], 4),
    metric_value=round(metrics[0], 4),
    clusters_count=3
)

dbscan_data = Algorithms(
    name="DBSCAN",
    clusters=dbscan_n_clusters,
    silhouette_score=round(dbscan_sil, 4),
    davies_bouldin=round(dbscan_db, 4),
    calisnky_harabasz=round(dbscan_ch, 4),
    score=round(best_scores[1], 4),
    metric_value=round(metrics[1], 4),
    clusters_count=dbscan_n_clusters,
    eps=0.7,
    samples=5,
    outliers=dbscan_outliers
)

new_benchmark = Statistics(
    date=timestamp,
    algorithms=[kmeans_data, dbscan_data],
    best_algorithm=best_algorithm
)


import os
json_path = 'benchmark/benchmark.json'
if os.path.exists(json_path):
    with open(json_path, "r") as f:
        existing_data = json.load(f)
    found = False
    for i, entry in enumerate(existing_data if isinstance(existing_data, list) else [existing_data]):
        if entry.get("date") == timestamp:
            existing_data[i] = asdict(new_benchmark)
            found = True
            break
    if not found:
        existing_data.append(asdict(new_benchmark))
    with open(json_path, "w") as f:
        json.dump(existing_data, f, indent=4)
else:
    with open(json_path, "w") as f:
        json.dump([asdict(new_benchmark)], f, indent=4)


print(f'DBSCAN clusters (excluindo ruído): {dbscan_n_clusters}')
print('\n--- Scores ---')
print('Silhouette:', metrics)
print('Normalized Scores:', best_scores)
print('Best Algorithm:', best_algorithm)
print(f"Benchmark salvo em {json_path}")
