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
                        
        # result_kmeans = group_kmeans(X_scaled)
        # Chamando a função de medição e passando o algoritmo
        result_kmeans = medir_performance(group_kmeans, X_scaled)
        print("\n")
        print("**** K-Means Execution ****")
        print(f"Tempo de execução: {result_kmeans['tempo_execucao']:.8f} segundos")
        print(f"Uso de memória: {result_kmeans['uso_memoria_MB']:.8f} MB")
        print(f"Uso médio de CPU: {result_kmeans['uso_cpu_percent']:.8f}%")
        
        
        # ====================== AGRUPAMENTO COM DBSCAN ======================================
        result_dbscan = medir_performance(group_dbscan, X_scaled)
        print("\n")
        print("**** DBSCAN Execution ****")
        print(f"Tempo de execução: {result_dbscan['tempo_execucao']:.8f} segundos")
        print(f"Uso de memória: {result_dbscan['uso_memoria_MB']:.8f} MB")
        print(f"Uso médio de CPU: {result_dbscan['uso_cpu_percent']:.8f}%")
        
        # return (result_kmeans['resultado'],result_dbscan['resultado'])
        return (result_kmeans,result_dbscan)
        
        
def calculate_metrics(result_kmeans, result_dbscan):
                   
        # ====================== CALCÚLO MÉTRICAS ============================================
        
        silhouette_norm_km = normalizeSilhouette(result_kmeans[0][0])
        silhouette_norm_dbs = normalizeSilhouette(result_dbscan[0][0])
        
        davies_norm = normalizeDavies(result_kmeans[0][1], result_dbscan[0][1])
        # calinski_norm
        
        davies_norm_km = davies_norm[0]
        davies_norm_dbs = davies_norm[1]
        # calinski_norm_km = normalize_calinski(result_kmeans[0][2])        
        # calinski_norm_dbs = normalize_calinski(result_dbscan[0][2])
        
        print("\n")
        print("***** Results Kmeans *****")
        print("Número de Clusters K-Means: ", result_kmeans[1])
        print('Objetos por cluster: ', result_kmeans[2][2])
        print("Melhores Resultados Kmeans (Silhouete, Davies_Bouldin, Calinski_Harabasz):", result_kmeans)  
        print("Melhores Resultados Kmeans Normalizados (Silhouete, Davies_Bouldin, Calinski_Harabasz):", silhouette_norm_km, davies_norm_km)  
                   
        print("\n")        
        print("***** Results DBSCAN *****")
	# result =  ((silhouette, davies_bouldin, calinski_harabasz), (n_clusters, n_outliers,cluster_counts), (eps, min_samples) )
        # result =  ((0.26976905276031005, 1.5007555451992962, 49.594810183036635), (2, 0, {0: 301, 1: 35}), (3.0, 3))
        print("Número de Clusters DBSCAN", result_dbscan[1][0])
        print("Melhores parâmetros (eps, samples):", result_dbscan[2])
        print("Melhores Resultados DBScan (Silhouete, Davies_Bouldin, Calinski_Harabasz, n_clusters, n_outliers):", result_dbscan[0])                        
        print("Melhores Resultados DBScan Normalizados (Silhouete, Davies_Bouldin, Calinski_Harabasz):", silhouette_norm_dbs, davies_norm_dbs)  
                
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
    eps_values = np.linspace(0.5, 5, 10) 
    min_samples_values = range(3, 10)
    best_params = dbscan.find_best_params(X_scaled, eps_values, min_samples_values)

    # Realizar o agrupamento
    best_results, params = dbscan.run (X_scaled, best_params[0], best_params[1]) 
    
    # best_results = dbscan.run (X_scaled, 0.5, 3) 
    return (best_results, params, best_params)

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
    qtd = "7"
    
    # Criando a instância do FileManager
    file_manager = FileManager("results/statistic_results.json")

    # Carregando os dados existentes do arquivo JSON
    results = file_manager.load_results()

    print("=========== HEALTH DOMAIN ==========\n")
    
    health_domain = Domain(name="Health")
    health_stat = Statistics(qtd_data=qtd)
    
    
    dados = medir_performance(process_data, webid, sensorType_health, qtd)
    print("\n")
    print("**** PREPROCESS TIME ****")
    print(f"Tempo de execução: {dados['tempo_execucao']:.8f} segundos")
    print(f"Uso de memória: {dados['uso_memoria_MB']:.8f} MB")
    print(f"Uso médio de CPU: {dados['uso_cpu_percent']:.8f}%")     
    
    health_stat.preprocess_time_s = dados['tempo_execucao']
    health_stat.preprocess_memo_mb = dados['uso_memoria_MB']
    health_stat.preprocess_cpu_perc = dados['uso_cpu_percent']

    # Envolve cálculo de paramêtros, execução do algoritmo e cálculo dos indices (silhouette, davies e calinski)
    results_all = run_algorithms(dados['resultado'])
    
    algo_km = Algorithms(name="Kmeans")
    results_kmeans = results_all[0]
    results_kmeans_metrics = results_kmeans ['resultado'][0]
    kmeans_labels = results_kmeans ['resultado'][3]
    # params_kmeans = results_kmeans ['resultado'][2]
    algo_km.clusters = results_kmeans['resultado'][1]    
    algo_km.silhouette_score = results_kmeans_metrics[0]
    algo_km.davies_bouldin = results_kmeans_metrics[1]
    algo_km.calisnky_harabasz = results_kmeans_metrics[2]
    algo_km.time_s = results_kmeans ['tempo_execucao']
    algo_km.memory_mb = results_kmeans['uso_memoria_MB']
    algo_km.cpu_perc = results_kmeans ['uso_cpu_percent']
    algo_km.clusters_count = results_kmeans['resultado'][2]
    # algo_km.clusters_count = params_kmeans[2]
    
    health_stat.algorithms.append(algo_km)
                
    algo_dbs = Algorithms(name="DBSCAN")
    results_dbscan = results_all[1]
    results_dbscan_metrics = results_dbscan ['resultado'][0]
    params_dbscan = results_dbscan ['resultado'][1]
    best_params = results_dbscan ['resultado'][2]
    algo_dbs.clusters = params_dbscan[0]
    algo_dbs.silhouette_score = results_dbscan_metrics[0]
    algo_dbs.davies_bouldin = results_dbscan_metrics[1]
    algo_dbs.calisnky_harabasz = results_dbscan_metrics[2]
    algo_dbs.time_s = results_dbscan ['tempo_execucao']
    algo_dbs.memory_mb = results_dbscan['uso_memoria_MB']
    algo_dbs.cpu_perc = results_dbscan ['uso_cpu_percent']
    algo_dbs.outliers = params_dbscan[1]
    algo_dbs.clusters_count = params_dbscan[2]
    algo_dbs.eps = best_params[0]
    algo_dbs.samples = best_params[1]

        
    health_stat.algorithms.append(algo_dbs)
    
    health_stat.run_time_s = algo_km.time_s + algo_dbs.time_s
    health_stat.run_memo_mb = algo_km.memory_mb + algo_dbs.memory_mb
    health_stat.run_cpu_perc = algo_km.cpu_perc + algo_dbs.cpu_perc
    
    # Usa os índices calculados anteriormente para gerar as métricas
    dados3 = medir_performance(calculate_metrics, results_kmeans ['resultado'], results_dbscan ['resultado'])
    print("\n")
    print("**** CALCULATE METRICS TIME ****")
    print(f"Tempo de execução: {dados3['tempo_execucao']:.8f} segundos")
    print(f"Uso de memória: {dados3['uso_memoria_MB']:.8f} MB")
    print(f"Uso médio de CPU: {dados3['uso_cpu_percent']:.8f}%") 
    metrics = dados3['resultado']
    
    health_stat.metrics_time_s = dados3['tempo_execucao']
    health_stat.metrics_memo_mb = dados3['uso_memoria_MB']
    health_stat.metrics_cpu_perc = dados3['uso_cpu_percent']

    dados4 = medir_performance(save_metrics, metrics[0], metrics[1], "Health")
    print("\n")
    print("**** SAVE TIME ****")
    print(f"Tempo de execução: {dados4['tempo_execucao']:.8f} segundos")
    print(f"Uso de memória: {dados4['uso_memoria_MB']:.8f} MB")
    print(f"Uso médio de CPU: {dados4['uso_cpu_percent']:.8f}%") 
    
    health_stat.save_time_s = dados4['tempo_execucao']
    health_stat.save_memo_mb = dados4['uso_memoria_MB']
    health_stat.save_cpu_perc = dados4['uso_cpu_percent']
    
    health_stat.total_time_s = health_stat.preprocess_time_s + health_stat.run_time_s + health_stat.metrics_time_s + health_stat.save_time_s
    health_stat.total_memo_mb = health_stat.preprocess_memo_mb + health_stat.run_memo_mb + health_stat.metrics_memo_mb + health_stat.save_memo_mb
    health_stat.total_cpu_perc = health_stat.preprocess_cpu_perc + health_stat.run_cpu_perc + health_stat.metrics_cpu_perc + health_stat.save_cpu_perc
    
    # Verificar se o domínio já existe
    domain = next((d for d in results.domains if d.name == health_domain.name), None)
    
    if not domain:
        # Se o domínio não existir, criar e adicionar
        domain = Domain(name="Health", statistics=[health_stat])
        results.domains.append(domain)
    else:
        # Verificar se a estatística já existe pelo qtd_data
        stat = next((s for s in domain.statistics if s.qtd_data == health_stat.qtd_data), None)
        
        if stat:
            # Atualizar estatística existente
            stat.__dict__.update(health_stat.__dict__)
        else:
            # Adicionar nova estatística
            domain.statistics.append(health_stat)
    
    title = "Health Domain with K-Means - "+qtd+" days"        
    shared.plot_pca(X_scaled=dados['resultado'], labels=kmeans_labels, file_name="results/"+qtd+"/"+qtd+"_km_pca_health.png", title=title)
    title = "Health Domain with DBSCAN - "+qtd+" days"        
    shared.plot_pca(X_scaled=dados['resultado'], labels=params_dbscan[3], file_name="results/"+qtd+"/"+qtd+"_dbscan_pca_health.png", title=title)
               
               
    print("=========== ENVIRONMENT DOMAIN ==========\n")
    env_domain = Domain(name="Environment")
    env_stat = Statistics(qtd_data=qtd)
    
    dados = medir_performance(process_data, webid, sensorType_env, qtd)
    print("\n")
    print("**** PREPROCESS TIME ****")
    print(f"Tempo de execução: {dados['tempo_execucao']:.8f} segundos")
    print(f"Uso de memória: {dados['uso_memoria_MB']:.8f} MB")
    print(f"Uso médio de CPU: {dados['uso_cpu_percent']:.8f}%")     

    env_stat.preprocess_time_s = dados['tempo_execucao']
    env_stat.preprocess_memo_mb = dados['uso_memoria_MB']
    env_stat.preprocess_cpu_perc = dados['uso_cpu_percent']
    
    # Envolve cálculo de paramêtros, execução do algoritmo e cálculo dos indices (silhouette, davies e calinski)
    results_all = run_algorithms(dados['resultado'])
    
    algo_km = Algorithms(name="Kmeans")
    results_kmeans = results_all[0]
    results_kmeans_metrics = results_kmeans ['resultado'][0]
    kmeans_labels = results_kmeans ['resultado'][3]
    kmeans_centroids = results_kmeans ['resultado'][4]
    algo_km.clusters = results_kmeans['resultado'][1]    
    algo_km.silhouette_score = results_kmeans_metrics[0]
    algo_km.davies_bouldin = results_kmeans_metrics[1]
    algo_km.calisnky_harabasz = results_kmeans_metrics[2]
    algo_km.time_s = results_kmeans ['tempo_execucao']
    algo_km.memory_mb = results_kmeans['uso_memoria_MB']
    algo_km.cpu_perc = results_kmeans ['uso_cpu_percent']
    algo_km.clusters_count = results_kmeans['resultado'][2]
    
    env_stat.algorithms.append(algo_km)
                
    algo_dbs = Algorithms(name="DBSCAN")
    results_dbscan = results_all[1]
    results_dbscan_metrics = results_dbscan ['resultado'][0]
    params_dbscan = results_dbscan ['resultado'][1]
    best_params = results_dbscan ['resultado'][2]
    algo_dbs.clusters = params_dbscan[0]
    algo_dbs.silhouette_score = results_dbscan_metrics[0]
    algo_dbs.davies_bouldin = results_dbscan_metrics[1]
    algo_dbs.calisnky_harabasz = results_dbscan_metrics[2]
    algo_dbs.time_s = results_dbscan ['tempo_execucao']
    algo_dbs.memory_mb = results_dbscan['uso_memoria_MB']
    algo_dbs.cpu_perc = results_dbscan ['uso_cpu_percent']
    algo_dbs.outliers = params_dbscan[1]
    algo_dbs.clusters_count = params_dbscan[2]
    algo_dbs.eps = best_params[0]
    algo_dbs.samples = best_params[1]

    env_stat.algorithms.append(algo_dbs)
    
    env_stat.run_time_s = algo_km.time_s + algo_dbs.time_s
    env_stat.run_memo_mb = algo_km.memory_mb + algo_dbs.memory_mb
    env_stat.run_cpu_perc = algo_km.cpu_perc + algo_dbs.cpu_perc
    
    # Usa os índices calculados anteriormente para gerar as métricas
    dados3 = medir_performance(calculate_metrics, results_kmeans ['resultado'], results_dbscan ['resultado'])
    print("\n")
    print("**** CALCULATE METRICS TIME ****")
    print(f"Tempo de execução: {dados3['tempo_execucao']:.8f} segundos")
    print(f"Uso de memória: {dados3['uso_memoria_MB']:.8f} MB")
    print(f"Uso médio de CPU: {dados3['uso_cpu_percent']:.8f}%") 
    metrics = dados3['resultado']
    
    env_stat.metrics_time_s = dados3['tempo_execucao']
    env_stat.metrics_memo_mb = dados3['uso_memoria_MB']
    env_stat.metrics_cpu_perc = dados3['uso_cpu_percent']
    
    dados4 = medir_performance(save_metrics, metrics[0], metrics[1], "Environment")
    print("\n")
    print("**** SAVE TIME ****")
    print(f"Tempo de execução: {dados4['tempo_execucao']:.8f} segundos")
    print(f"Uso de memória: {dados4['uso_memoria_MB']:.8f} MB")
    print(f"Uso médio de CPU: {dados4['uso_cpu_percent']:.8f}%")
    
    env_stat.save_time_s = dados4['tempo_execucao']
    env_stat.save_memo_mb = dados4['uso_memoria_MB']
    env_stat.save_cpu_perc = dados4['uso_cpu_percent']
    
    env_stat.total_time_s = env_stat.preprocess_time_s + env_stat.run_time_s + env_stat.metrics_time_s + env_stat.save_time_s
    env_stat.total_memo_mb = env_stat.preprocess_memo_mb + env_stat.run_memo_mb + env_stat.metrics_memo_mb + env_stat.save_memo_mb
    env_stat.total_cpu_perc = env_stat.preprocess_cpu_perc + env_stat.run_cpu_perc + env_stat.metrics_cpu_perc + env_stat.save_cpu_perc

 
    # Verificar se o domínio já existe
    domain = next((d for d in results.domains if d.name == env_domain.name), None)
    
    if not domain:
        # Se o domínio não existir, criar e adicionar
        domain = Domain(name="Environment", statistics=[env_stat])
        results.domains.append(domain)
    else:
        # Verificar se a estatística já existe pelo qtd_data
        stat = next((s for s in domain.statistics if s.qtd_data == env_stat.qtd_data), None)
        
        if stat:
            # Atualizar estatística existente
            stat.__dict__.update(env_stat.__dict__)
        else:
            # Adicionar nova estatística
            domain.statistics.append(env_stat)
 
    title = "Environment Domain with K-Means - "+qtd+" days"        
    shared.plot_graph_kmeans(X=dados['resultado'], labels=kmeans_labels, centroids=kmeans_centroids, file_name="results/"+qtd+"/"+qtd+"_kmeans_graph_environment.png", title=title)
    title = "Environment Domain with DBSCAN - "+qtd+" days"        
    shared.plot_dbscan_clusters(X=dados['resultado'], labels=params_dbscan[3], file_name="results/"+qtd+"/"+qtd+"_dbscan_graph_environment.png", title=title)
 
 
    print("=========== ENVIRONMENT_HEALTH DOMAIN ==========\n")   
    env_health_domain = Domain(name="Environment_Health")
    env_health_stat = Statistics(qtd_data=qtd)
    
    dados = medir_performance(process_data, webid, sensorType_all, qtd)
    print("\n")
    print("**** PREPROCESS TIME ****")
    print(f"Tempo de execução: {dados['tempo_execucao']:.8f} segundos")
    print(f"Uso de memória: {dados['uso_memoria_MB']:.8f} MB")
    print(f"Uso médio de CPU: {dados['uso_cpu_percent']:.8f}%")     

    env_health_stat.preprocess_time_s = dados['tempo_execucao']
    env_health_stat.preprocess_memo_mb = dados['uso_memoria_MB']
    env_health_stat.preprocess_cpu_perc = dados['uso_cpu_percent']
           
    # Envolve cálculo de paramêtros, execução do algoritmo e cálculo dos indices (silhouette, davies e calinski)
    results_all = run_algorithms(dados['resultado'])
    
    algo_km = Algorithms(name="Kmeans")
    results_kmeans = results_all[0]
    results_kmeans_metrics = results_kmeans ['resultado'][0]
    algo_km.clusters = results_kmeans['resultado'][1]    
    kmeans_labels = results_kmeans ['resultado'][3]
    algo_km.silhouette_score = results_kmeans_metrics[0]
    algo_km.davies_bouldin = results_kmeans_metrics[1]
    algo_km.calisnky_harabasz = results_kmeans_metrics[2]
    algo_km.time_s = results_kmeans ['tempo_execucao']
    algo_km.memory_mb = results_kmeans['uso_memoria_MB']
    algo_km.cpu_perc = results_kmeans ['uso_cpu_percent']
    algo_km.clusters_count = results_kmeans['resultado'][2]
    
    env_health_stat.algorithms.append(algo_km)
                
    algo_dbs = Algorithms(name="DBSCAN")
    results_dbscan = results_all[1]
    results_dbscan_metrics = results_dbscan ['resultado'][0]
    params_dbscan = results_dbscan ['resultado'][1]
    best_params = results_dbscan ['resultado'][2]
    algo_dbs.clusters = params_dbscan[0]
    algo_dbs.silhouette_score = results_dbscan_metrics[0]
    algo_dbs.davies_bouldin = results_dbscan_metrics[1]
    algo_dbs.calisnky_harabasz = results_dbscan_metrics[2]
    algo_dbs.time_s = results_dbscan ['tempo_execucao']
    algo_dbs.memory_mb = results_dbscan['uso_memoria_MB']
    algo_dbs.cpu_perc = results_dbscan ['uso_cpu_percent']
    algo_dbs.outliers = params_dbscan[1]
    algo_dbs.clusters_count = params_dbscan[2]
    algo_dbs.eps = best_params[0]
    algo_dbs.samples = best_params[1]

        
    env_health_stat.algorithms.append(algo_dbs)
    
    env_health_stat.run_time_s = algo_km.time_s + algo_dbs.time_s
    env_health_stat.run_memo_mb = algo_km.memory_mb + algo_dbs.memory_mb
    env_health_stat.run_cpu_perc = algo_km.cpu_perc + algo_dbs.cpu_perc
    
    # Usa os índices calculados anteriormente para gerar as métricas
    dados3 = medir_performance(calculate_metrics, results_kmeans ['resultado'], results_dbscan ['resultado'])
    print("\n")
    print("**** CALCULATE METRICS TIME ****")
    print(f"Tempo de execução: {dados3['tempo_execucao']:.8f} segundos")
    print(f"Uso de memória: {dados3['uso_memoria_MB']:.8f} MB")
    print(f"Uso médio de CPU: {dados3['uso_cpu_percent']:.8f}%") 
    metrics = dados3['resultado']    
        
    env_health_stat.metrics_time_s = dados3['tempo_execucao']
    env_health_stat.metrics_memo_mb = dados3['uso_memoria_MB']
    env_health_stat.metrics_cpu_perc = dados3['uso_cpu_percent']
    
    # gc.collect()
    dados4 = medir_performance(save_metrics, metrics[0], metrics[1], "Environment_Health")
    print("\n")
    print("**** SAVE TIME ****")
    print(f"Tempo de execução: {dados4['tempo_execucao']:.8f} segundos")
    print(f"Uso de memória: {dados4['uso_memoria_MB']:.8f} MB")
    print(f"Uso médio de CPU: {dados4['uso_cpu_percent']:.8f}%")
    
    env_health_stat.save_time_s = dados4['tempo_execucao']
    env_health_stat.save_memo_mb = dados4['uso_memoria_MB']
    env_health_stat.save_cpu_perc = dados4['uso_cpu_percent']
    
    env_health_stat.total_time_s = env_health_stat.preprocess_time_s + env_health_stat.run_time_s + env_health_stat.metrics_time_s + env_health_stat.save_time_s
    env_health_stat.total_memo_mb = env_health_stat.preprocess_memo_mb + env_health_stat.run_memo_mb + env_health_stat.metrics_memo_mb + env_health_stat.save_memo_mb
    env_health_stat.total_cpu_perc = env_health_stat.preprocess_cpu_perc + env_health_stat.run_cpu_perc + env_health_stat.metrics_cpu_perc + env_health_stat.save_cpu_perc
    
    # Verificar se o domínio já existe
    domain = next((d for d in results.domains if d.name == env_health_domain.name), None)
    
    if not domain:
        # Se o domínio não existir, criar e adicionar
        domain = Domain(name="Environment_Health", statistics=[env_health_stat])
        results.domains.append(domain)
    else:
        # Verificar se a estatística já existe pelo qtd_data
        stat = next((s for s in domain.statistics if s.qtd_data == env_health_stat.qtd_data), None)
        
        if stat:
            # Atualizar estatística existente
            stat.__dict__.update(env_health_stat.__dict__)
        else:
            # Adicionar nova estatística
            domain.statistics.append(env_health_stat)
    
    title = "Environment and Health Domains with K-Means - "+qtd+" days"        
    shared.plot_pca(X_scaled=dados['resultado'], labels=kmeans_labels, file_name="results/"+qtd+"/"+qtd+"_km_pca_environment_health.png", title=title)
    title = "Environment and Health Domains with DBSCAN - "+qtd+" days"        
    shared.plot_pca(X_scaled=dados['resultado'], labels=params_dbscan[3], file_name="results/"+qtd+"/"+qtd+"_dbscan_pca_environment_health.png", title=title)
    
    print("\nSalvando em JSON.")
    print ("========================================\n")
    
    file_manager.save_results(results)
    
    print("\nExecução concluída.")
    print ("========================================\n")
    
