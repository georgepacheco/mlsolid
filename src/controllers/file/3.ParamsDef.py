import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from datetime import datetime

# --------------------------
# ETAPA 0 - Carregamento e Pré-processamento
# --------------------------
data_atual = datetime.now().strftime("%Y%m%d")
input_path = os.path.join("dataset", f"dataset_{data_atual}.csv")

output_dir = os.path.join("dataset", data_atual)
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_path)  # ajuste o caminho conforme necessário
df_clean = df.dropna()

scaler = StandardScaler()
X = scaler.fit_transform(df_clean)

# --------------------------
# ETAPA 1 - PCA para Visualização
# --------------------------
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)
pca_df = pd.DataFrame(pca_result, columns=["PCA1", "PCA2"])

plt.figure(figsize=(8, 6))
sns.scatterplot(x="PCA1", y="PCA2", data=pca_df, alpha=0.6)
plt.title("Visualização PCA (2 Componentes)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"bench_pca_{data_atual}.png"))
plt.close()


# --------------------------
# ETAPA 2 - Método do Cotovelo (K-Means)
# --------------------------
inertia = []
k_values = range(1, 10)

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o')
plt.title("Método do Cotovelo (K-Means)")
plt.xlabel("Número de Clusters (k)")
plt.ylabel("Inércia")
plt.xticks(k_values)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"bench_elbow_{data_atual}.png"))
plt.close()

# --------------------------
# ETAPA 3 - Gráfico da Distância k-Vizinhos (DBSCAN)
# --------------------------
min_samples = 5
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(X)
distances, indices = neighbors_fit.kneighbors(X)
k_distances = np.sort(distances[:, -1])

plt.figure(figsize=(8, 5))
plt.plot(k_distances)
plt.title(f"Gráfico da Distância k-Vizinhos (k={min_samples}) para DBSCAN")
plt.xlabel("Pontos ordenados")
plt.ylabel(f"Distância para o {min_samples}º vizinho")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"bench_kneighbors_{data_atual}.png"))
plt.close()

