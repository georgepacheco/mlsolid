from clusters import dbscan
import numpy as np
import pandas as pd

#  DATASET INTELAB ======================================================================
# Lista com os nomes das colunas
# collumns = ['date', 'time', 'epoch', 'moteid', 'temperature', 'humidity', 'light', 'voltage']

# # Carregar dados
# df = pd.read_csv("../data.txt", sep=" ", engine='python', names=collumns)
# df_1000 = df.iloc[:10000].copy()

# # Remove Collumns
# df2 = df_1000.drop(columns=['date','time', 'epoch', 'moteid',])

# DATASET SAÚDE =========================================================================
# collumns = ['Age','Blood Glucose Level(BGL)','Diastolic Blood Pressure','Systolic Blood Pressure','Heart Rate','Body Temperature','SPO2','Sweating  (Y/N)','Shivering (Y/N)','Diabetic/NonDiabetic (D/N)']
    
# # Carregar dados
# df = pd.read_csv("data/pt9.csv", sep=",", engine='python', names=collumns)

# # Remove Collumns
# df2 = df.drop(columns=['Diabetic/NonDiabetic (D/N)'])

# df2 = df2.apply(pd.to_numeric, errors='coerce')
# # print (df2)

# DATASET CONJUNTO =========================================================================        
# collumns = ['Age','Blood Glucose Level(BGL)','Diastolic Blood Pressure','Systolic Blood Pressure','Heart Rate','Body Temperature','SPO2','Sweating  (Y/N)','Shivering (Y/N)','Diabetic/NonDiabetic (D/N)']

# Carregar dados
df = pd.read_csv("data/health_intel.csv", sep=",", engine='python')

# Remove Collumns
df2 = df.drop(columns=['Age', 'Diabetic/NonDiabetic (D/N)'])

df2 = df2.apply(pd.to_numeric, errors='coerce')
# print (df2.info())


# PROCESSAMENTO =========================================================================
X_scaled = dbscan.preprocess(df2)
# print (X_scaled)

# Testar combinações de parâmetros
eps_values = np.linspace(0.5, 5, 10)
# eps_values = np.linspace(0.2, 5, 25)
# print (eps_values)
min_samples_values = range(3, 10)

best_params, best_results = dbscan.find_best_params(X_scaled, eps_values, min_samples_values)
    
# best_params, best_results =   dbscan.find_auto_params(X_scaled, eps_values, min_samples_values)

print("Melhores parâmetros (eps, samples, clusters, outliers):", best_params)
print("Melhores Resultados (Silhouete, Davies_Bouldin, Calinski_Harabasz):", best_results)

