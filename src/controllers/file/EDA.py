import matplotlib.pyplot as plt
import seaborn as sns

# 1. Resumo estatístico
summary_stats = df.describe()

# 2. Histogramas das variáveis
df.hist(bins=20, figsize=(12, 8), edgecolor='black')
plt.suptitle('Distribuição das Variáveis', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()

# 3. Boxplots para detecção de outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.title("Boxplots das Variáveis")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Matriz de correlação
plt.figure(figsize=(8, 6))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlação")
plt.tight_layout()
plt.show()

# Exibir resumo estatístico como tabela
import ace_tools as tools; tools.display_dataframe_to_user(name="Resumo Estatístico", dataframe=summary_stats)

