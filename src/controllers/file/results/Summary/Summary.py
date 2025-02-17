import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_file(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def time_memory_graph (data):
    records = []
    for domain in data["domains"]:
        domain_name = domain["name"]
        for stat in domain["statistics"]:
            records.append({
                "Domain": domain_name,
                "qtd_data": int(stat["qtd_data"]),
                "total_time": round(float(stat["total_time_s"]),2),
                "total_memo": round(float(stat["total_memo_mb"]),2)
            })
    
    # Criar DataFrame
    df = pd.DataFrame(records)
    
    df["Domain"] = pd.Categorical(df["Domain"], categories=[d["name"] for d in data["domains"]], ordered=True)
    df = df.sort_values(by=["Domain", "qtd_data"])
    
    df.to_excel("spreadsheets/total_memo_times.xlsx", index=False)
    # Criar gráfico
    plt.figure(figsize=(10, 6))
    for domain in df["Domain"].unique():
        subset = df[df["Domain"] == domain].sort_values(by="qtd_data")           
        plt.plot(subset["qtd_data"], subset["total_time"], marker='o', label=domain)

            
    # Personalização do gráfico
    # Definir os valores do eixo X como os valores únicos de qtd_data
    unique_qtd_data = sorted(df["qtd_data"].unique())  # Ordenar para garantir a sequência correta
    plt.xticks(unique_qtd_data)
    plt.xlabel("Data (days)")
    plt.ylabel("Time (s)")
    # plt.title("Clustering execution time by domain")
    plt.legend()
    plt.grid(True)
    # file_name = f"graphs/{domain}_time.png"
    file_name = "graphs/total_time.png"
    plt.savefig(file_name)
    plt.close()
    
    for domain in df["Domain"].unique():
        subset = df[df["Domain"] == domain].sort_values(by="qtd_data")           
        plt.plot(subset["qtd_data"], subset["total_memo"], marker='o', label=domain)

    # Personalização do gráfico
    # Definir os valores do eixo X como os valores únicos de qtd_data
    unique_qtd_data = sorted(df["qtd_data"].unique())  # Ordenar para garantir a sequência correta
    plt.xticks(unique_qtd_data)
    plt.xlabel("Data (days)")
    plt.ylabel("Memmory (MB)")
    # plt.title("Memory consumption of clustering by domain")
    plt.legend()
    plt.grid(True)
    # file_name = f"graphs/{domain}_time.png"
    file_name = "graphs/total_memory.png"
    plt.savefig(file_name)
    plt.close()

def algo_time_memory_graph(data):
    # Extrair os dados
    records = []
    for domain in data["domains"]:
        domain_name = domain["name"]
        for stat in domain["statistics"]:
            for algo in stat["algorithms"]:
                records.append({
                    "Domain": domain_name,
                    "qtd_data": int(stat["qtd_data"]),
                    "Algorithm": algo["name"],
                    "time_s": round(float(algo["time_s"] if algo["time_s"] is not None else 0),2),
                    "memory_mb": round(float(algo["memory_mb"] if algo["memory_mb"] is not None else 0),2)
                })

    # Criar DataFrame
    df = pd.DataFrame(records)   
    
    df["Domain"] = pd.Categorical(df["Domain"], categories=[d["name"] for d in data["domains"]], ordered=True)
    df = df.sort_values(by=["Domain", "Algorithm", "qtd_data"])
     
    df.to_excel("spreadsheets/algo_memo_times.xlsx", index=False)
    
        
    for algo in df["Algorithm"].unique():
        plt.figure(figsize=(10, 6))
        
        subset = df[df["Algorithm"] == algo]
        for domain in subset["Domain"].unique():
            domain_subset = subset[subset["Domain"] == domain]
            plt.plot(domain_subset["qtd_data"], domain_subset["time_s"], marker='o', label=domain)
        
        # Definir os valores do eixo X como os valores únicos de qtd_data
        unique_qtd_data = sorted(df["qtd_data"].unique())  # Ordenar para garantir a sequência correta
        plt.xticks(unique_qtd_data)
        
        plt.xlabel("Data (days)")
        plt.ylabel("Time (ms)")
        plt.title(f"Execution time for Algorithm - {algo}")
        plt.legend(title="Domains")
        plt.grid()
        
        file_name = f"graphs/{algo}_time.png"
        plt.savefig(file_name)
        plt.close()
                   
    for algo in df["Algorithm"].unique():
        plt.figure(figsize=(10, 6))
        
        subset = df[df["Algorithm"] == algo]
        for domain in subset["Domain"].unique():
            domain_subset = subset[subset["Domain"] == domain]
            plt.plot(domain_subset["qtd_data"], domain_subset["memory_mb"], marker='o', label=domain)
        
        # Definir os valores do eixo X como os valores únicos de qtd_data
        unique_qtd_data = sorted(df["qtd_data"].unique())  # Ordenar para garantir a sequência correta
        plt.xticks(unique_qtd_data)
        
        plt.xlabel("Data (days)")
        plt.ylabel("Memory (MB)")
        # plt.title(f"Execution time for Algorithm - {algo}")
        plt.legend(title="Domains")
        plt.grid()
        
        file_name = f"graphs/{algo}_memo.png"
        plt.savefig(file_name)
        plt.close()
        
def metrics_graph(data):
    # Extrair os dados
    records = []
    for domain in data["domains"]:
        domain_name = domain["name"]
        for stat in domain["statistics"]:
            for algo in stat["algorithms"]:
                records.append({
                    "Domain": domain_name,
                    "qtd_data": int(stat["qtd_data"]),
                    "Algorithm": algo["name"],                    
                    "Silhouette": algo["silhouette_score"] if algo["silhouette_score"] is not None else 0,
                    "Davies-Bouldin": algo["davies_bouldin"] if algo["davies_bouldin"] is not None else 0,
                    "Calinski-Harabasz": algo["calisnky_harabasz"] if algo["calisnky_harabasz"] is not None else 0
                })

    # # Criar DataFrame
    # df = pd.DataFrame(records)    
    # df.to_csv("spreadsheets/metrics.csv", index=False)
    
    # normalized_dfs = []
    # for domain, group in df.groupby("Domain"):
    #     group["Silhouette_Norm"] = min_max_normalize(group["Silhouette"])
        
    #     # Invertendo Davies-Bouldin (menor é melhor) antes de normalizar
    #     group["Davies-Bouldin_Inverted"] = group["Davies-Bouldin"].max() - group["Davies-Bouldin"]
    #     group["Davies-Bouldin_Norm"] = min_max_normalize(group["Davies-Bouldin_Inverted"])
        
    #     group["Calinski-Harabasz_Norm"] = min_max_normalize(group["Calinski-Harabasz"])
        
    #     # Removendo a coluna intermediária invertida
    #     group.drop(columns=["Davies-Bouldin_Inverted", "Silhouette", "Davies-Bouldin", "Calinski-Harabasz"], inplace=True)
        
    #     normalized_dfs.append(group)
    # # Concatenando os resultados normalizados
    # df_normalized = pd.concat(normalized_dfs)

    # # Salvando os dados normalizados em um CSV
    # df_normalized.to_csv("spreadsheets/metrics_normalized.csv", index=False)
    
    # print(df_normalized.head())
    
    # Criar DataFrame
    df = pd.DataFrame(records)

    # Inverter Davies-Bouldin antes da normalização (menor é melhor)
    df["Davies-Bouldin"] = df["Davies-Bouldin"].max() - df["Davies-Bouldin"]

    # Aplicar logaritmo ao Calinski-Harabasz para reduzir escala
    df["Calinski-Harabasz"] = np.log1p(df["Calinski-Harabasz"])  # log1p(x) = log(1 + x), evita problemas com valores 0

    # Normalizar todas as métricas JUNTAS dentro de cada Domain
    normalized_dfs = []
    for domain, group in df.groupby("Domain"):
        metrics = ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"]
        group[metrics] = normalize_all(group, metrics)
        normalized_dfs.append(group)

    # Concatenar os resultados normalizados
    df_normalized = pd.concat(normalized_dfs)

    # Salvar os dados normalizados
    df_normalized.to_csv("spreadsheets/metrics_normalized_combined.csv", index=False)

    # Exibir os primeiros resultados
    print(df_normalized.head())
    
    # # Criar gráficos de comparação das métricas
    # metrics = ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"]
    # for domain in df["Domain"].unique():
    #     subset = df[df["Domain"] == domain]
    #     qtd_data_values = sorted(subset["qtd_data"].unique())
    #     x = np.arange(len(qtd_data_values))
    #     width = 0.3
        
    #     for metric in metrics:
    #         plt.figure(figsize=(12, 6))
    #         for i, algo in enumerate(subset["Algorithm"].unique()):
    #             algo_values = [subset[(subset["Algorithm"] == algo) & (subset["qtd_data"] == q)][metric].values[0] if not subset[(subset["Algorithm"] == algo) & (subset["qtd_data"] == q)].empty else 0 for q in qtd_data_values]
    #             plt.bar(x + i * width, algo_values, width, label=algo)
            
    #         plt.xlabel("Qtd Data")
    #         plt.ylabel(metric)
    #         plt.title(f"{metric} Score Comparison - {domain}")
    #         plt.xticks(x + width / 2, qtd_data_values)
    #         plt.legend()
    #         plt.grid(axis='y')
            
    #         file_name = f"graphs/{domain}_{metric}_comparison.png"
    #         plt.savefig(file_name)
    #         plt.close() 

def metrics_domain_table (data):
    # Extrair os dados
    records = []
    for domain in data["domains"]:
        domain_name = domain["name"]
        for stat in domain["statistics"]:
            for algo in stat["algorithms"]:
                records.append({
                    "Domain": domain_name,
                    "qtd_data": int(stat["qtd_data"]),
                    "Algorithm": algo["name"],                    
                    "Silhouette": round(float(algo["silhouette_score"] if algo["silhouette_score"] is not None else 0),2),
                    "Davies-Bouldin": round(float(algo["davies_bouldin"] if algo["davies_bouldin"] is not None else 0),2),
                    "Calinski-Harabasz": round(float(algo["calisnky_harabasz"] if algo["calisnky_harabasz"] is not None else 0),2)
                })
                
    # Criar DataFrame
    df = pd.DataFrame(records)   
    
    df["Domain"] = pd.Categorical(df["Domain"], categories=[d["name"] for d in data["domains"]], ordered=True)
    df = df.sort_values(by=["Domain", "qtd_data"])
         
    df.to_excel("spreadsheets/metrics_domain.xlsx", index=False)      
    
         

def min_max_normalize(series):
    min_val = series.min()
    max_val = series.max()
    return (series - min_val) / (max_val - min_val) if max_val > min_val else series            

def normalize_all(df, columns):
    min_val = df[columns].min().min()
    max_val = df[columns].max().max()
    return (df[columns] - min_val) / (max_val - min_val) if max_val > min_val else df[columns]
