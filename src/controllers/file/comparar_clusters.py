
# ===============================================
# COMPARAÇÃO ENTRE CLUSTER BENCHMARK E AUTOMÁTICO
# ===============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix

# ============================
# 1. CARREGAMENTO DOS DADOS
# ============================
# Substitua os caminhos pelos seus arquivos reais
df_benchmark = pd.read_csv("benchmark_clusters.csv")           # Deve conter a coluna 'Cluster'
df_automatico = pd.read_csv("kmeans_result_auto.csv")    # Deve conter a coluna 'Cluster'

# ============================
# 2. COMBINAÇÃO DOS RÓTULOS
# ============================
# Supondo que ambas as tabelas têm a mesma ordem
df_benchmark["Cluster_Automatico"] = df_automatico["Cluster"]

# ============================
# 3. CÁLCULO DAS MÉTRICAS
# ============================
ari = adjusted_rand_score(df_benchmark["Cluster"], df_benchmark["Cluster_Automatico"])
nmi = normalized_mutual_info_score(df_benchmark["Cluster"], df_benchmark["Cluster_Automatico"])

print(f"Adjusted Rand Index (ARI): {ari:.3f}")
print(f"Normalized Mutual Information (NMI): {nmi:.3f}")

# ============================
# 4. MATRIZ DE CONFUSÃO
# ============================
conf_mat = confusion_matrix(df_benchmark["Cluster"], df_benchmark["Cluster_Automatico"])

plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusão entre Benchmark e Método Automático")
plt.xlabel("Automático")
plt.ylabel("Benchmark")
plt.tight_layout()
plt.show()
