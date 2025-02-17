import json
import tempfile
import subprocess
import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import MinMaxScaler
from Performance import medir_performance
from dataclasses import asdict
from Model import FileManager, Results, Domain, Statistics, Algorithms

# Adiciona o diretório raiz do projeto ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from clusters import kmeans
from clusters import shared
from clusters import dbscan

def prepare_data(file_path):
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
        
        return X_scaled

def run_algorithms(X_scaled):
       # ====================== AGRUPAMENTO COM KMEANS ======================================                        
        result_kmeans = group_kmeans(X_scaled)
        
        # ====================== AGRUPAMENTO COM DBSCAN ======================================
        result_dbscan = group_dbscan(X_scaled)
        
        
        return(result_kmeans, result_dbscan)
        
        
def calculate_metrics(result_kmeans, result_dbscan):
                   
        # ====================== CALCÚLO MÉTRICAS ============================================
        
        silhouette_norm_km = normalizeSilhouette(result_kmeans[0][0])
        
        if result_dbscan[0][0] != None:
            silhouette_norm_dbs = normalizeSilhouette(result_dbscan[0][0])
        
        
        # davies_norm = normalizeDavies(result_kmeans[0][1], result_dbscan[0][1])
        # calinski_norm
        
        # davies_norm_km = davies_norm[0]
        # davies_norm_dbs = davies_norm[1]
        # calinski_norm_km = normalize_calinski(result_kmeans[0][2])        
        # calinski_norm_dbs = normalize_calinski(result_dbscan[0][2])
        
        print("\n")
        print("***** Results Kmeans *****")
        print("Número de Clusters K-Means: ", result_kmeans[1])
        print('Objetos por cluster: ', result_kmeans[2])
        print("Melhores Resultados Kmeans (Silhouete, Davies_Bouldin, Calinski_Harabasz):", result_kmeans[0])  
        print("Melhores Resultados Kmeans Normalizados (Silhouete, Davies_Bouldin, Calinski_Harabasz):", silhouette_norm_km)  
                   
        print("\n")        
        print("***** Results DBSCAN *****")
	# result =  ((silhouette, davies_bouldin, calinski_harabasz), (n_clusters, n_outliers,cluster_counts), (eps, min_samples) )
        # result =  ((0.26976905276031005, 1.5007555451992962, 49.594810183036635), (2, 0, {0: 301, 1: 35}), (3.0, 3))
        print("Número de Clusters DBSCAN", result_dbscan[1][0])
        print("Melhores parâmetros (eps, samples):", result_dbscan[2])
        print("Melhores Resultados DBScan (Silhouete, Davies_Bouldin, Calinski_Harabasz, n_clusters, n_outliers):", result_dbscan[0])                        
        print("Melhores Resultados DBScan Normalizados (Silhouete, Davies_Bouldin, Calinski_Harabasz):", silhouette_norm_dbs)  
                
        return (silhouette_norm_km, silhouette_norm_dbs)
                

def save_metrics(kmeans_metric, dbscan_metric, domain):
    # ====================== SAVE RESULTS INTO SOLID ======================================        
        # enviar o melhor valor para o solid
        if (kmeans_metric > dbscan_metric):
            # Executa o script Node.js para gerar os dados
            # print ('Enviando ... ', str(result_kmeans))
            subprocess.run(["node", "../../../dist/controllers/file/SaveMetrics.js", webid, str(kmeans_metric), domain], check=True)
        else:
            # Executa o script Node.js para gerar os dados
            # print ('Enviando ... ', str(result_dbscan[0]))
            subprocess.run(["node", "../../../dist/controllers/file/SaveMetrics.js", webid, str(dbscan_metric), domain], check=True)
    
       
def group_kmeans(X_scaled):
    # Calculate the optimal k        
    optimal_k = kmeans.elbow(X_scaled)
    # print ("Clusters: ", optimal_k)
    
    # Run kmeans
    results, params = kmeans.run_kmeans(X_scaled, optimal_k)
    
    # retorna indices, n_clusters, params = (kmeans.labels_, kmeans.cluster_centers_, cluster_counts)
    return (results, optimal_k, params[2], params[0], params[1])
    
    # # retorna indices, n_clusters, params = (kmeans.labels_, kmeans.cluster_centers_, cluster_counts)
    # return (results, optimal_k, params[2])
    
def group_dbscan(X_scaled):
     
    # Definir melhores parametros
    # eps_values = np.linspace(0.5, 5, 10) 
    # min_samples_values = range(3, 10)
    #best_params = dbscan.find_best_params(X_scaled, eps_values, min_samples_values)
    #print ("Best Params Velho - Group DBSCAN (eps, min): ", best_params)
    # print ("Velho método: ", best_params)
    
    min = dbscan.estimate_min_samples(X_scaled)
    eps = dbscan.find_optimal_epsilon(X_scaled, min)
    best_params = (eps, min)
    print ("Best Params Novo - Group DBSCAN (eps, min): ", eps, min)
    
    best_results, params = dbscan.run (X_scaled, best_params[0], best_params[1])
    
    return (best_results, params, best_params)
    # if best_params != None:	   
    #     best_results, params = dbscan.run (X_scaled, best_params[0], best_params[1])
    #     return (best_results, params, best_params)
    # return None

# normalize to [0,1]            
def normalizeSilhouette(value):             
    return (value + 1) / 2;

# normalize to [0,1]                
def normalizeDavies (km_value, dbs_value):
    # Valores fictícios de Davies-Bouldin
    db_values = [km_value, dbs_value]  

    # Passo 1: Inverter os valores
    db_inverted = [1/x for x in db_values]

    # Passo 2: Criar o scaler
    scaler = MinMaxScaler()

    # Passo 3: Normalizar os valores
    db_normalized = scaler.fit_transform(np.array(db_inverted).reshape(-1, 1)).flatten()

    return db_normalized

# normalize to [0,1]            
def normalize_calinski(km_value, dbs_value):
    print ("normalize calinski")


def process_data (webid, sensorType, limit): 
    # Arquivo temporário para comunicação
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
        temp_file_path = temp_file.name
        
        try:
            # Converte a lista de sensores em uma string separada por vírgulas
            sensor_types_str = ",".join(sensorType)
                       
            # Executa o script Node.js para gerar os dados
            subprocess.run(["node", "../../../dist/controllers/file/GenerateFile.js", temp_file_path, webid, sensor_types_str, limit], check=True)

            # Consome os dados gerados pelo Node.js
            X_scaled = prepare_data(temp_file_path)

        except subprocess.CalledProcessError as e:
            print("Erro ao executar o script Node.js:", e)

        finally:
            # Remove o arquivo temporário, se necessário
            import os
            os.remove(temp_file_path)
            print("Arquivo temporário removido.")
        return X_scaled

if __name__ == "__main__":
    
    print("Iniciando execução do Consumer...")
    webid = "https://192.168.0.111:3000/Joao/profile/card#me"
    sensorType_health = ["Glucometer", "HeartBeatSensor", "SystolicBloodPressure", "DiastolicBloodPressure", "BodyThermometer", "SkinConductanceSensor", "Accelerometer", "PulseOxymeter"]
    sensorType_env = ["AirThermometer", "HumiditySensor"]
   # sensorType_all = ["AirThermometer", "HumiditySensor","Glucometer", "HeartBeatSensor", "BloodPressureSensor", "BodyThermometer", "SkinConductanceSensor", "Accelerometer", "PulseOxymeter"]
    sensorType_all = ["AirThermometer", "HumiditySensor","Glucometer", "HeartBeatSensor", "SystolicBloodPressure", "DiastolicBloodPressure", "BodyThermometer", "SkinConductanceSensor", "Accelerometer", "PulseOxymeter"]
    qtd = "90"
    

    print("=========== HEALTH DOMAIN ==========\n")
        
    dados_health = process_data(webid, sensorType_health, qtd)
    
    # Envolve cálculo de paramêtros, execução do algoritmo e cálculo dos indices (silhouette, davies e calinski)
    results_all_health = run_algorithms(dados_health)
       
    metrics_health = calculate_metrics (results_all_health[0], results_all_health[1])
   

    save_metrics (metrics_health[0], metrics_health[1], "Health")

    print("=========== ENVIRONMENT DOMAIN ==========\n")
    
           
    dados_env = process_data(webid, sensorType_env, qtd)
    
    # Envolve cálculo de paramêtros, execução do algoritmo e cálculo dos indices (silhouette, davies e calinski)
    results_all_env = run_algorithms(dados_env)
       
    metrics_env = calculate_metrics (results_all_env[0], results_all_env[1])
   

    save_metrics (metrics_env[0], metrics_env[1], "Environment")
    
    print("=========== ENVIRONMENT ALL ==========\n")
           
    dados_env_health = process_data(webid, sensorType_all, qtd)
    
    # Envolve cálculo de paramêtros, execução do algoritmo e cálculo dos indices (silhouette, davies e calinski)
    results_all_env_health = run_algorithms(dados_env_health)
       
    metrics_env_health = calculate_metrics (results_all_env_health[0], results_all_env_health[1])
   

    save_metrics (metrics_env_health[0], metrics_env_health[1], "Env_Health")
    
    print("\nExecução concluída.")
    print ("========================================\n")
    
