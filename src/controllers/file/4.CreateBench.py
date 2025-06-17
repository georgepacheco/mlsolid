import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# --------------------------
# ETAPA 0 - Preparação e Carregamento
# --------------------------

data_atual = datetime.now().strftime("%Y%m%d")

output_dir = os.path.join("benchmark", data_atual)
os.makedirs(output_dir, exist_ok=True)

input_path = os.path.join("dataset", f"dataset_{data_atual}.csv")

df = pd.read_csv(input_path)
print ("DF len", len(df))
#df_clean = df.dropna()
df_clean = df.fillna(df.mean())

scaler = StandardScaler()
X = scaler.fit_transform(df_clean)

# PCA para visualização
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)
pca_df = pd.DataFrame(pca_result, columns=["PCA1", "PCA2"])

# --------------------------
# ETAPA 1 - Clustering
# --------------------------

# --- K-Means ---
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
kmeans_labels = kmeans.fit_predict(X)
kmeans_score = silhouette_score(X, kmeans_labels)
kmeans_davies = davies_bouldin_score(X, kmeans_labels)
kmeans_calinski = calinski_harabasz_score(X, kmeans_labels)
kmeans_clusters = len(set(kmeans_labels))

# --- DBSCAN ---
dbscan = DBSCAN(eps=0.8, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)
core_mask = dbscan_labels != -1
dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
dbscan_outliers = list(dbscan_labels).count(-1)

if dbscan_clusters > 0 and core_mask.sum() > 1:
    dbscan_score = silhouette_score(X[core_mask], dbscan_labels[core_mask])
    dbscan_davies = davies_bouldin_score(X[core_mask], dbscan_labels[core_mask])
    dbscan_calinski = calinski_harabasz_score(X[core_mask], dbscan_labels[core_mask])
else:
    dbscan_score = None
    dbscan_davies = None
    dbscan_calinski = None

# Adiciona os rótulos
pca_df["KMeans"] = kmeans_labels
pca_df["DBSCAN"] = dbscan_labels

# --------------------------
# ETAPA 2 - Visualização
# --------------------------

# K-Means
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x="PCA1", y="PCA2", hue="KMeans", palette="viridis")
#plt.title(f"K-Means Clustering (k=3) - Silhouette Score: {kmeans_score:.2f}")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"bench_pca_kmeans_{data_atual}.png"))
plt.close()

# DBSCAN
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x="PCA1", y="PCA2", hue="DBSCAN", palette="tab10")
#plt.title(f"DBSCAN Clustering (eps=0.8) - Silhouette Score: {dbscan_score:.2f}" if dbscan_score else "DBSCAN Clustering (eps=0.8)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"bench_pca_dbscan_{data_atual}.png"))
plt.close()

# --------------------------
# ETAPA 3 - Análise Estatística
# --------------------------

# K-Means
#df_kmeans = df_clean.copy()
df_kmeans = pd.DataFrame(X)
df_kmeans["KMeans_Cluster"] = kmeans_labels
stats_kmeans = df_kmeans.groupby("KMeans_Cluster").mean().round(2)
print("\nEstatísticas por Cluster (K-Means):")
print(stats_kmeans)

# DBSCAN
#df_dbscan = df_clean.copy()
df_dbscan = pd.DataFrame(X)
df_dbscan["DBSCAN_Cluster"] = dbscan_labels
df_core = df_dbscan[core_mask]
stats_dbscan = df_core.groupby("DBSCAN_Cluster").mean().round(2)
print("\nEstatísticas por Cluster (DBSCAN):")
print(stats_dbscan)

# --------------------------
# ETAPA 4 - Exportação de Dados
# --------------------------

# Dataset com rótulos do K-Means
df_kmeans.to_csv(os.path.join(output_dir, "kmeans_result_benchmark.csv"), index=False)

# Dataset com rótulos do DBSCAN
df_dbscan.to_csv(os.path.join(output_dir, "dbscan_result_benchmark.csv"), index=False)

print(len(df_kmeans))

# --------------------------
# ETAPA 5 - Salvando benchmark.json
# --------------------------

benchmark_json_path = os.path.join("benchmark", "benchmark.json")
benchmark_entry = {
    "date": int(data_atual),
    "algorithms": [
        {
            "name": "KMeans",
            "clusters": 3,
            "silhouette_score": round(kmeans_score, 4),
            "davies_bouldin": round(kmeans_davies, 4),
            "calisnky_harabasz": round(kmeans_calinski, 2),
            "clusters_count": kmeans_clusters
        },
        {
            "name": "DBSCAN",
            "eps": 0.8,
            "samples": 5,
            "silhouette_score": round(dbscan_score, 4) if dbscan_score else None,
            "davies_bouldin": round(dbscan_davies, 4) if dbscan_davies else None,
            "calisnky_harabasz": round(dbscan_calinski, 2) if dbscan_calinski else None,
            "clusters_count": dbscan_clusters,
            "outliers": dbscan_outliers
        }
    ]
}

# Carrega os benchmarks existentes
if os.path.exists(benchmark_json_path):
    with open(benchmark_json_path, "r") as f:
        try:
            all_benchmarks = json.load(f)
        except json.JSONDecodeError:
            all_benchmarks = []
else:
    all_benchmarks = []

# Garante que o conteúdo seja uma lista
if isinstance(all_benchmarks, dict):
    all_benchmarks = [all_benchmarks]

# Remove entrada antiga da mesma data, se existir
all_benchmarks = [entry for entry in all_benchmarks if entry.get("date") != int(data_atual)]

# Adiciona a nova entrada
all_benchmarks.append(benchmark_entry)

# Salva o arquivo atualizado
with open(benchmark_json_path, "w") as f:
    json.dump(all_benchmarks, f, indent=4)
# --------------------------
# ETAPA FINAL - Mensagem de Sucesso
# --------------------------

print(f"\nArquivos salvos com sucesso em: {output_dir}")
print(f"Benchmark atualizado: {benchmark_json_path}")

