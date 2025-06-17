import json
import tempfile
import subprocess
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from Performance import medir_performance
from dataclasses import asdict, dataclass, field
from typing import List, Optional
from Model import FileManager, Results, Domain, Statistics, Algorithms
from datetime import datetime
from pathlib import Path
from sklearn.decomposition import PCA


# Adiciona o diretório raiz do projeto ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from clusters import kmeans
from clusters import shared
from clusters import dbscan

@dataclass
class Algo:
    name: Optional[str] = None
    clusters: Optional[int] = None
    silhouette_score: Optional[float] = None
    davies_bouldin: Optional[float] = None
    calisnky_harabasz: Optional[float] = None
    score: Optional[float] = None
    metric_value: Optional[float] = None
    time_s: Optional[float] = None
    memory_mb: Optional[float] = None
    cpu_perc: Optional[float] = None
    clusters_count: Optional[int] = None
    eps: Optional[float] = None
    samples: Optional[int] = None
    outliers: Optional[int] = None
    

@dataclass
class Stat:
    qtd_data: Optional[int] = None
    index: Optional[int] = None
    norm_data: Optional[str] = None
    norm2_metric: Optional[str] = None
    best_algorithm: Optional[str] = None
    preprocess_time_s: Optional[float] = None
    preprocess_memo_mb: Optional[float] = None
    preprocess_cpu_perc: Optional[float] = None
    run_time_s: Optional[float] = None
    run_memo_mb: Optional[float] = None
    run_cpu_perc: Optional[float] = None
    metrics_time_s: Optional[float] = None
    metrics_memo_mb: Optional[float] = None
    metrics_cpu_perc: Optional[float] = None
    save_time_s: Optional[float] = None
    save_memo_mb: Optional[float] = None
    save_cpu_perc: Optional[float] = None
    total_time_s: Optional[float] = None
    total_memo_mb: Optional[float] = None
    total_cpu_perc: Optional[float] = None
    algorithms: List[Algorithms] = field(default_factory=list)

def prepare_data(file_path, norm):
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
        X_scaled = shared.preprocess(df, norm)
        
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
        if result_dbscan['resultado'] != None:
            print("\n")
            print("**** DBSCAN Execution ****")
            print(f"Tempo de execução: {result_dbscan['tempo_execucao']:.8f} segundos")
            print(f"Uso de memória: {result_dbscan['uso_memoria_MB']:.8f} MB")
            print(f"Uso médio de CPU: {result_dbscan['uso_cpu_percent']:.8f}%")
        
            # return (result_kmeans['resultado'],result_dbscan['resultado'])
            return (result_kmeans,result_dbscan)
        return(result_kmeans, None)
        
        
def calculate_metrics(result_kmeans, result_dbscan):
                   
        # ====================== CALCÚLO MÉTRICAS ============================================
        
        silhouette_norm_km = normalizeSilhouette(result_kmeans[0][0])
        
        if result_dbscan[0][0] != None:
            silhouette_norm_dbs = normalizeSilhouette(result_dbscan[0][0])        
        
        print("\n")
        print("***** Results Kmeans *****")
        print("Número de Clusters K-Means: ", result_kmeans[1])
        print('Objetos por cluster: ', result_kmeans[2][2])
        print("Melhores Resultados Kmeans (Silhouete, Davies_Bouldin, Calinski_Harabasz):", result_kmeans)  
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
                

def save_metrics(metric, domain):
    # ====================== SAVE RESULTS INTO SOLID ======================================        
    subprocess.run(["node", "../../../dist/controllers/file/SaveMetrics.js", webid, str(metric), domain], check=True)       
    
       
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
       
    min = dbscan.estimate_min_samples(X_scaled)
    eps = dbscan.find_optimal_epsilon(X_scaled, min)
    best_params = (eps, min)
    print ("Best Params Novo - Group DBSCAN (eps, min): ", eps, min)
    
    
    if best_params != None:	   
        best_results, params = dbscan.run (X_scaled, best_params[0], best_params[1])
        return (best_results, params, best_params)
    return None

# normalize to [0,1]            
def normalizeSilhouette(value):             
    return (value + 1) / 2;

# normalize to [0,1]                
def normalize_dbi(value):
    """Inverte e normaliza Davies-Bouldin Index (menor é melhor)."""
    return 1 / (1 + value)  

def normalize_chi(value, max_chi):
    """Aplica normalização logarítmica para Calinski-Harabasz Index."""    
    return np.log1p(value) / np.log1p(max_chi)

def custom_norm (silhouette_kmeans, silhouette_dbscan, dbi_kmeans, dbi_dbscan, chi_kmeans, chi_dbscan):
    # Normalizando as métricas
    sil_kmeans_norm = normalizeSilhouette(silhouette_kmeans)
    sil_dbscan_norm = normalizeSilhouette(silhouette_dbscan)
    print ("Silhouette: ", sil_kmeans_norm, sil_dbscan_norm)

    dbi_kmeans_norm = normalize_dbi(dbi_kmeans)
    dbi_dbscan_norm = normalize_dbi(dbi_dbscan)
    print("DBI: ",dbi_kmeans_norm, dbi_dbscan_norm)

    max_chi = max(chi_kmeans, chi_dbscan)
    chi_kmeans_norm = normalize_chi(chi_kmeans, max_chi)
    chi_dbscan_norm = normalize_chi(chi_dbscan, max_chi)
    print ("Chi: ", chi_kmeans_norm, chi_dbscan_norm)
    
    return (sil_kmeans_norm, dbi_kmeans_norm, chi_kmeans_norm, sil_dbscan_norm, dbi_dbscan_norm, chi_dbscan_norm)

def min_max_norm (silhouette_kmeans, silhouette_dbscan, dbi_kmeans, dbi_dbscan, chi_kmeans, chi_dbscan):
    # Silhouette
    sil_min = min(silhouette_kmeans, silhouette_dbscan)
    sil_max = max(silhouette_kmeans, silhouette_dbscan)
    sil_kmeans_norm = (silhouette_kmeans - sil_min) / (sil_max - sil_min) if sil_max != sil_min else 0.5
    sil_dbscan_norm = (silhouette_dbscan - sil_min) / (sil_max - sil_min) if sil_max != sil_min else 0.5

    # DBI (quanto menor, melhor) – inverte o valor antes da normalização
    dbi_kmeans_inv = -dbi_kmeans
    dbi_dbscan_inv = -dbi_dbscan
    dbi_min = min(dbi_kmeans_inv, dbi_dbscan_inv)
    dbi_max = max(dbi_kmeans_inv, dbi_dbscan_inv)
    dbi_kmeans_norm = (dbi_kmeans_inv - dbi_min) / (dbi_max - dbi_min) if dbi_max != dbi_min else 0.5
    dbi_dbscan_norm = (dbi_dbscan_inv - dbi_min) / (dbi_max - dbi_min) if dbi_max != dbi_min else 0.5

    # CHI (quanto maior, melhor)
    chi_min = min(chi_kmeans, chi_dbscan)
    chi_max = max(chi_kmeans, chi_dbscan)
    chi_kmeans_norm = (chi_kmeans - chi_min) / (chi_max - chi_min) if chi_max != chi_min else 0.5
    chi_dbscan_norm = (chi_dbscan - chi_min) / (chi_max - chi_min) if chi_max != chi_min else 0.5

    # Logs de debug (opcional)
    print("Silhouette:", sil_kmeans_norm, sil_dbscan_norm)
    print("DBI:", dbi_kmeans_norm, dbi_dbscan_norm)
    print("CHI:", chi_kmeans_norm, chi_dbscan_norm)

    return (sil_kmeans_norm, dbi_kmeans_norm, chi_kmeans_norm, sil_dbscan_norm, dbi_dbscan_norm, chi_dbscan_norm)

def choose_best_algorithm(silhouette_kmeans, silhouette_dbscan, dbi_kmeans, dbi_dbscan, chi_kmeans, chi_dbscan, norm='custom'):
    """
    Calcula o melhor algoritmo baseado nas métricas normalizadas.
    """
    
    if (norm == 'custom' ):          
        indices_norm = custom_norm (silhouette_kmeans, silhouette_dbscan, dbi_kmeans, dbi_dbscan, chi_kmeans, chi_dbscan)
    elif (norm == 'minmax'):
        indices_norm = min_max_norm (silhouette_kmeans, silhouette_dbscan, dbi_kmeans, dbi_dbscan, chi_kmeans, chi_dbscan)
 
    score_kmeans = (indices_norm[0] + indices_norm[1] + indices_norm[2])/3
    score_dbscan = (indices_norm[3] + indices_norm[4] + indices_norm[5])/3 	       
    
    # Escolhendo o melhor algoritmo
    # best_algorithm = "K-Means" if score_kmeans > score_dbscan else "DBSCAN"
    if (score_kmeans > score_dbscan):
        best_algorithm = "K-Means"
        best_metric = indices_norm[0]
    else:
        best_algorithm = "DBSCAN"
        best_metric = indices_norm[3]
    
    best_scores = (score_kmeans, score_dbscan)  # Agora garantidamente entre 0 e 1
    metrics = (indices_norm[0], indices_norm[3])

    return best_algorithm, best_scores, metrics, best_metric

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
  
def run_experiment (norm_data, data_qtd, norm_metric, replica):                 
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
    
    # Salvar rótulos automáticos em CSV para comparação com benchmark
    # df_kmeans_auto = pd.DataFrame({"Cluster": kmeans_labels})
    df_kmeans_auto = pd.DataFrame(dados['resultado'])
    df_kmeans_auto["KMeans_Cluster"] = kmeans_labels
    output_kmeans_path = f"results/{qtd}/kmeans_result_auto.csv"
    df_kmeans_auto.to_csv(output_kmeans_path, index=False)
    print(len(df_kmeans_auto))
    
    kmeans_centroids = results_kmeans ['resultado'][4]
    algo_km.clusters = results_kmeans['resultado'][1]    
    algo_km.silhouette_score = results_kmeans_metrics[0]
    algo_km.davies_bouldin = results_kmeans_metrics[1]
    algo_km.calisnky_harabasz = results_kmeans_metrics[2]
    algo_km.time_s = results_kmeans ['tempo_execucao']
    algo_km.memory_mb = results_kmeans['uso_memoria_MB']
    algo_km.cpu_perc = results_kmeans ['uso_cpu_percent']
    algo_km.clusters_count = results_kmeans['resultado'][2]
    
    # env_stat.algorithms.append(algo_km)
    
    algo_dbs = Algorithms(name="DBSCAN")
    if results_all[1] != None:                    
        results_dbscan = results_all[1]
        results_dbscan_metrics = results_dbscan ['resultado'][0]
        params_dbscan = results_dbscan ['resultado'][1]        
        dbscan_labels = params_dbscan[3]  # Certifique-se que está correto
        
        
        df_dbscan_auto = pd.DataFrame(dados['resultado'])
        df_dbscan_auto["DBSCAN_Cluster"] = dbscan_labels        
        output_dbscan_path = f"results/{qtd}/dbscan_result_auto.csv"
        df_dbscan_auto.to_csv(output_dbscan_path, index=False)
        print(f"Resultado do DBSCAN salvo para comparação: {output_dbscan_path}")
        
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
        env_stat.run_time_s = algo_km.time_s + algo_dbs.time_s
        env_stat.run_memo_mb = algo_km.memory_mb + algo_dbs.memory_mb
        env_stat.run_cpu_perc = algo_km.cpu_perc + algo_dbs.cpu_perc
    else:
        algo_dbs.clusters = None
        algo_dbs.silhouette_score = None
        algo_dbs.davies_bouldin = None
        algo_dbs.calisnky_harabasz = None
        algo_dbs.time_s = None
        algo_dbs.memory_mb = None
        algo_dbs.cpu_perc = None
        algo_dbs.outliers = None
        algo_dbs.clusters_count = None
        algo_dbs.eps = None
        algo_dbs.samples = None
        env_stat.run_time_s = 0
        env_stat.run_memo_mb = 0
        env_stat.run_cpu_perc = 0

    # env_stat.algorithms.append(algo_dbs)
    
    km = results_kmeans['resultado']      
    dbs = results_dbscan['resultado']     
    # Usa os índices calculados anteriormente para gerar as métricas
    # dados3 = medir_performance(calculate_metrics, results_kmeans ['resultado'], results_dbscan ['resultado'])
    dados3 = medir_performance(choose_best_algorithm, km[0][0], dbs[0][0], km[0][1], dbs[0][1], km[0][2], dbs[0][2])
    print("\n")
    print("**** CALCULATE METRICS TIME ****")
    print(f"Tempo de execução: {dados3['tempo_execucao']:.8f} segundos")
    print(f"Uso de memória: {dados3['uso_memoria_MB']:.8f} MB")
    print(f"Uso médio de CPU: {dados3['uso_cpu_percent']:.8f}%") 
    # best_algorithm, best_scores, metrics, best_metric
    metrics = dados3['resultado']
    print ("Metricas Calculadas: ", metrics)
    
    env_stat.metrics_time_s = dados3['tempo_execucao']
    env_stat.metrics_memo_mb = dados3['uso_memoria_MB']
    env_stat.metrics_cpu_perc = dados3['uso_cpu_percent']
    
    env_stat.best_algorithm = metrics[0]
    
    algo_km.score = metrics[1][0]
    algo_km.metric_value = metrics[2][0]
    
    algo_dbs.score = metrics[1][1]
    algo_dbs.metric_value = metrics[2][1]
          
    env_stat.algorithms.append(algo_km)
    env_stat.algorithms.append(algo_dbs)
    
    dados4 = medir_performance(save_metrics, metrics[3], "Environment")
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

    
    # --------------------------
    # ETAPA 2 - Visualização
    # --------------------------
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(dados['resultado'])
    pca_df = pd.DataFrame(pca_result, columns=["PCA1", "PCA2"])
    
    # Adiciona os rótulos
    pca_df["KMeans"] = kmeans_labels
    pca_df["DBSCAN"] = dbscan_labels
    
    
    output_dir = os.path.join("results", data_atual)
    os.makedirs(output_dir, exist_ok=True)
    
    # K-Means
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x="PCA1", y="PCA2", hue="KMeans", palette="viridis")
    #plt.title(f"K-Means Clustering (k=3) - Silhouette Score: {kmeans_score:.2f}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"auto_pca_kmeans_{data_atual}.png"))
    plt.close()

    # DBSCAN
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x="PCA1", y="PCA2", hue="DBSCAN", palette="tab10")
    #plt.title(f"DBSCAN Clustering (eps=0.8) - Silhouette Score: {dbscan_score:.2f}" if dbscan_score else "DBSCAN Clustering (eps=0.8)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"auto_pca_dbscan_{data_atual}.png"))
    plt.close()    

             
    print("\nSalvando em JSON.")
    print ("========================================\n")
    
    file_manager.save_results(results)
    
    print("\nExecução concluída.")
    print ("========================================\n")
    
if __name__ == "__main__":
    
    print("Iniciando execução do Consumer...")
    webid = "https://192.168.0.111:3000/Joao/profile/card#me"
    
    sensorType_env = ["HumiditySensor", "AirThermometer", "CO_Sensor", "LightSensor"]

    
    # Pega a data atual para usar como qtd
    data_atual = datetime.now().strftime("%Y%m%d")
    # Converte para o formato YYYYMMDD e transforma em inteiro
    qtd = data_atual

    # Caminho base: diretório onde está o script
    base_path = Path(__file__).parent

    # Caminho da pasta "results"
    results_path = base_path / "experiment3"

    # Nome do novo subdiretório
    novo_diretorio = results_path / qtd

    # Criação do diretório (e da pasta results se ainda não existir)
    novo_diretorio.mkdir(parents=True, exist_ok=True)
    
            
    # Criando a instância do FileManager
    file_manager = FileManager("experiment3/experiment_results.json")

    # Carregando os dados existentes do arquivo JSON
    results = file_manager.load_results()

    # Caminho para o CSV de planejamento
    PLAN_CSV_PATH = "design_fatorial.csv"
    
    # Carrega o planejamento experimental
    plan_df = pd.read_csv(PLAN_CSV_PATH)
    
    # Loop pelos experimentos
    for i, row in plan_df.iterrows():
        print(f"\n==== Executando experimento {i+1}/{len(plan_df)} - Ordem {row['run_order']} ====")
        try:
            run_experiment(
            norm_data=row['norm_data'],
            data_qtd=int(row['data']),
            norm_metric=row['norm_metric'],
            replica=int(row['replica']),
            run_order=int(row['run_order'])
        )
        except Exception as e:
            print(f"Erro ao executar a execução {i+1}: {e}")