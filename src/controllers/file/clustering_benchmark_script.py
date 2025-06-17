
# ==========================================
# PASSO A PASSO: CLUSTERING EXPLORATÓRIO
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

# ================================
# 1. CARREGAMENTO E LIMPEZA DOS DADOS
# ================================
df = pd.read_csv("meu_dataset.csv")
df_clean = df.dropna()

# ================================
# 2. ANÁLISE ESTATÍSTICA DESCRITIVA
# ================================
print(df_clean.describe())
df_clean.hist(bins=20, figsize=(12, 8), edgecolor='black')
plt.suptitle("Distribuição das Variáveis")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_clean)
plt.title("Boxplot das Variáveis")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df_clean.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlação")
plt.show()

# ================================
# 3. REDUÇÃO DE DIMENSIONALIDADE (PCA)
# ================================
X = df_clean.values
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=10, alpha=0.5)
plt.title("PCA - Visualização em 2D")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()

# ================================
# 4. ESTIMATIVA DA QUANTIDADE DE CLUSTERS
# ================================
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(8, 4))
plt.plot(K_range, inertias, marker='o')
plt.title("Elbow Method")
plt.xlabel("Número de Clusters (K)")
plt.ylabel("Inércia")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(K_range, silhouette_scores, marker='s', color='green')
plt.title("Silhouette Score")
plt.xlabel("Número de Clusters (K)")
plt.ylabel("Score")
plt.grid(True)
plt.show()

# ================================
# 5. APLICAÇÃO DO K-MEANS COM K=3 (EXEMPLO)
# ================================
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)
df_clean["Cluster"] = labels

# ================================
# 6. ANÁLISE DOS CLUSTERS FORMADOS
# ================================
cluster_summary = df_clean.groupby("Cluster").agg(["mean", "std", "min", "max"])
print(cluster_summary)

# Visualização no espaço PCA
df_clean["PCA1"] = X_pca[:, 0]
df_clean["PCA2"] = X_pca[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_clean, x="PCA1", y="PCA2", hue="Cluster", palette="viridis", s=20)
plt.title("Clusters no Espaço PCA (K=3)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.legend(title="Cluster")
plt.show()

# ================================
# 7. SALVAR RESULTADO COMO BENCHMARK
# ================================
df_clean.to_csv("benchmark_clusters.csv", index=False)
