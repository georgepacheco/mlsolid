import json
import tempfile
import subprocess
import pandas as pd
import numpy as np
import sys
import os

# Adiciona o diretório raiz do projeto ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from clusters import kmeans
from clusters import shared
from clusters import dbscan

def prepare_data(file_path, domain):
    # Lê os dados do arquivo
    with open(file_path, 'r') as file:
        data = json.load(file)        
                
        # Dicionário para armazenar os dados por tipo de sensor
        sensor_columns = {}
        
        # Processa os dados
        for sensor_data in data:
            sensor_type = sensor_data['sensorType']  
            if 'observation' in sensor_data:
                # Garante que a chave para esse tipo de sensor exista no dicionário
                if sensor_type not in sensor_columns:
                    sensor_columns[sensor_type] = []
                    
                # Adiciona os valores das observações
                for observation in sensor_data['observation']:
                    result_value = observation.get('resultValue', pd.NA)
                    sensor_columns[sensor_type].append(result_value)

        # Ajusta o tamanho das colunas para garantir alinhamento
        max_length = max(len(values) for values in sensor_columns.values())

        for sensor_type, values in sensor_columns.items():
            # Preenche com 'N/A' se houver colunas de tamanhos diferentes
            sensor_columns[sensor_type].extend([pd.NA] * (max_length - len(values)))
	
        # Criação do DataFrame com pandas
        df = pd.DataFrame(sensor_columns)
        print(df.info())
        # df.to_csv("meu_dataset.csv", ",", index=False)                
        
        # Preprocess
        X_scaled = shared.preprocess(df)
        
        # ====================== AGRUPAMENTO COM KMEANS ======================================
                        
        result_kmeans = group_kmeans(X_scaled)
        
        silhouette_norm_km = normalizeSilhouette(result_kmeans[0][0])
        davies_norm_km = normalizeDavies(result_kmeans[0][1])
        calinski_norm_km = normalize_calinski(result_kmeans[0][2])
        
        print("Número de Clusters K-Means", result_kmeans[1])
        print("Melhores Resultados Kmeans (Silhouete, Davies_Bouldin, Calinski_Harabasz):", result_kmeans)  
        print("Melhores Resultados Kmeans Normalizados (Silhouete, Davies_Bouldin, Calinski_Harabasz):", silhouette_norm_km, davies_norm_km, calinski_norm_km)  
        # ====================== AGRUPAMENTO COM DBSCAN ======================================
        
        result_dbscan = group_dbscan(X_scaled)
        silhouette_norm_dbs = normalizeSilhouette(result_dbscan[0][0])
        davies_norm_dbs = normalizeDavies(result_dbscan[0][1])
        calinski_norm_dbs = normalize_calinski(result_dbscan[0][2])
        
        print("Número de Cluesters DBSCAN", result_dbscan[0][3])
        print("Melhores parâmetros (eps, samples):", result_dbscan[1])
        print("Melhores Resultados DBScan (Silhouete, Davies_Bouldin, Calinski_Harabasz, n_clusters, n_outliers):", result_dbscan[0])                
        
        print("Melhores Resultados DBScan Normalizados (Silhouete, Davies_Bouldin, Calinski_Harabasz):", silhouette_norm_dbs, davies_norm_dbs, calinski_norm_dbs)  
        # ====================== SAVE RESULTS INTO SOLID ======================================
        
        # enviar o melhor valor para o solid
        if (silhouette_norm_km > silhouette_norm_dbs):
            # Executa o script Node.js para gerar os dados
            # print ('Enviando ... ', str(result_kmeans))
            subprocess.run(["node", "../../../dist/controllers/file/SaveMetrics.js", webid, str(silhouette_norm_km), domain], check=True)
        else:
            # Executa o script Node.js para gerar os dados
            # print ('Enviando ... ', str(result_dbscan[0]))
            subprocess.run(["node", "../../../dist/controllers/file/SaveMetrics.js", webid, str(silhouette_norm_dbs), domain], check=True)

        
        # # enviar o melhor valor para o solid
        # if (result_kmeans[0] > result_dbscan[0][0]):
        #     # Executa o script Node.js para gerar os dados
        #     # print ('Enviando ... ', str(result_kmeans))
        #     subprocess.run(["node", "../../../dist/controllers/file/SaveMetrics.js", webid, str(result_kmeans), domain], check=True)
        # else:
        #     # Executa o script Node.js para gerar os dados
        #     # print ('Enviando ... ', str(result_dbscan[0]))
        #     subprocess.run(["node", "../../../dist/controllers/file/SaveMetrics.js", webid, str(result_dbscan[0]), domain], check=True)
        
        
def group_kmeans(X_scaled):
    # Calculate the optimal k        
    optimal_k = kmeans.elbow(X_scaled)
    # print ("Clusters: ", optimal_k)
    
    # Run kmeans
    results, params = kmeans.run_kmeans(X_scaled, optimal_k)
    return (results, optimal_k)
    
def group_dbscan(X_scaled):
     
    # Definir melhores parametros
    eps_values = np.linspace(0.5, 5, 10) 
    min_samples_values = range(3, 10)
    best_params = dbscan.find_best_params(X_scaled, eps_values, min_samples_values)

    # Realizar o agrupamento
    best_results = dbscan.run (X_scaled, best_params[0], best_params[1]) 
    
    # best_results = dbscan.run (X_scaled, 0.5, 3) 
    return (best_results, best_params)

# normalize to [0,1]            
def normalizeSilhouette(value):             
    return (value + 1) / 2;

# normalize to [0,1]                
def normalizeDavies (value):
    print ("normalize davies")

# normalize to [0,1]            
def normalize_calinski(value):
    print ("normalize calinski")


def process_data (webid, sensorType, domain, limit): 
    # Arquivo temporário para comunicação
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
        temp_file_path = temp_file.name
        
        try:
            # Converte a lista de sensores em uma string separada por vírgulas
            sensor_types_str = ",".join(sensorType)
                       
            # Executa o script Node.js para gerar os dados
            subprocess.run(["node", "../../../dist/controllers/file/GenerateFile.js", temp_file_path, webid, sensor_types_str, limit], check=True)

            # Consome os dados gerados pelo Node.js
            prepare_data(temp_file_path, domain)

        except subprocess.CalledProcessError as e:
            print("Erro ao executar o script Node.js:", e)

        finally:
            # Remove o arquivo temporário, se necessário
            import os
            os.remove(temp_file_path)
            print("Arquivo temporário removido.")

if __name__ == "__main__":
    
    print("Iniciando execução do Consumer...")
    webid = "https://192.168.0.111:3000/Joao/profile/card#me"
    sensorType_health = ["Glucometer", "HeartBeatSensor", "SystolicBloodPressure", "DiastolicBloodPressure", "BodyThermometer", "SkinConductanceSensor", "Accelerometer", "PulseOxymeter"]
    sensorType_env = ["AirThermometer", "HumiditySensor"]
   # sensorType_all = ["AirThermometer", "HumiditySensor","Glucometer", "HeartBeatSensor", "BloodPressureSensor", "BodyThermometer", "SkinConductanceSensor", "Accelerometer", "PulseOxymeter"]
    sensorType_all = ["AirThermometer", "HumiditySensor","Glucometer", "HeartBeatSensor", "SystolicBloodPressure", "DiastolicBloodPressure", "BodyThermometer", "SkinConductanceSensor", "Accelerometer", "PulseOxymeter"]
    
    qtd = "48"
    
    process_data(webid, sensorType_health, "Health", qtd)  # Chama a função definida no consumer.py
    
    process_data(webid, sensorType_env, "Environment", qtd)
    
    process_data(webid, sensorType_all, "Environment_Health", qtd)
    
    print("Execução concluída.")
