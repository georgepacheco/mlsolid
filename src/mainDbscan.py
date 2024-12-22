from clusters import dbscan, shared
import numpy as np
import pandas as pd
from memory_profiler import profile

@profile
def run ():
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
    # df = pd.read_csv("meu_dataset.csv", sep=",", engine='python')
    df = pd.read_csv("data/health_intel.csv", sep=",", engine='python')
    df_replicado = pd.concat([df] * 4, ignore_index=True)

    # temp_cols = ['sn3_temp', 'sn12_temp', 'sn14_temp', 'sn37_temp']
    # humid_cols = ['sn3_humid', 'sn12_humid', 'sn14_humid', 'sn37_humid']

    # df_new = pd.DataFrame({
    #     'temp': df[temp_cols].stack().reset_index(drop=True),
    #     'humid': df[humid_cols].stack().reset_index(drop=True)
    # })

    # print (df_new)
    # print (df_new.info())

    # # Remove Collumns
    # # df2 = df.drop(columns=['Age', 'Diabetic/NonDiabetic (D/N)'])

    # df2 = df.apply(pd.to_numeric, errors='coerce')

    # # print (df2.info())


    # PROCESSAMENTO =========================================================================
    X_scaled = shared.preprocess(df_replicado)
    # print (X_scaled)

    # Testar combinações de parâmetros
    eps_values = np.linspace(0.5, 5, 10)
    min_samples_values = range(3, 10)
    best_params = dbscan.find_best_params(X_scaled, eps_values, min_samples_values)

    # Realizar o agrupamento
    best_results = dbscan.run (X_scaled, best_params[0], best_params[1]) 
            
    print("Melhores parâmetros (eps, samples):", best_params)
    print("Melhores Resultados DBScan (Silhouete, Davies_Bouldin, Calinski_Harabasz, n_clusters, n_outliers):", best_results)


if __name__ == "__main__":
    run()