import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score

def carregar_rotulos(path_auto, path_benchmark, algoritmo):
    df_auto = pd.read_csv(path_auto)
    df_bench = pd.read_csv(path_benchmark)

    if len(df_auto) != len(df_bench):
        raise ValueError("Os arquivos possuem números diferentes de amostras.")

    # Define o nome da coluna de acordo com o algoritmo
    coluna = "KMeans_Cluster" if algoritmo == "KMeans" else "DBSCAN_Cluster"

    return df_auto[coluna].values, df_bench[coluna].values


def comparar_clusters(r_auto, r_bench, algoritmo):
    # Remove outliers (rótulos -1) se houver
    mask = (r_auto != -1) & (r_bench != -1)
    r1 = np.array(r_auto)[mask]
    r2 = np.array(r_bench)[mask]

    print(f"\n Comparando {algoritmo} - {len(r1)} amostras válidas")

    return {
        "algorithm": algoritmo,
        "ARI": round(adjusted_rand_score(r1, r2), 4),
        "NMI": round(normalized_mutual_info_score(r1, r2), 4),
        "FMI": round(fowlkes_mallows_score(r1, r2), 4)
    }

def salvar_resultado(resultados, data_str):
    comparison_file = Path("results") / "comparison.json"

    nova_entrada = {
        "date": int(data_str),
        "comparisons": resultados
    }

    # Carregar conteúdo existente (tratando casos antigos)
    if comparison_file.exists():
        with open(comparison_file, "r") as f:
            try:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    all_comparisons = [loaded]
                else:
                    all_comparisons = loaded
            except json.JSONDecodeError:
                all_comparisons = []
    else:
        all_comparisons = []

    # Substituir se já houver entrada para a data
    all_comparisons = [entry for entry in all_comparisons if entry.get("date") != int(data_str)]
    all_comparisons.append(nova_entrada)

    # Salvar atualizado
    with open(comparison_file, "w") as f:
        json.dump(all_comparisons, f, indent=4)

    print(f"\n Resultados salvos em: {comparison_file}")

if __name__ == "__main__":
    data_str = datetime.now().strftime("%Y%m%d")
    base_path_auto = Path(f"results/{data_str}")
    base_path_bench = Path(f"benchmark/{data_str}")

    arquivos = {
        "KMeans": {
            "auto": base_path_auto / "kmeans_result_auto.csv",
            "bench": base_path_bench / "kmeans_result_benchmark.csv"
        },
        "DBSCAN": {
            "auto": base_path_auto / "dbscan_result_auto.csv",
            "bench": base_path_bench / "dbscan_result_benchmark.csv"
        }
    }

    resultados = []
    for algoritmo, caminhos in arquivos.items():
        try:
            r_auto, r_bench = carregar_rotulos(caminhos["auto"], caminhos["bench"], algoritmo)
            resultado = comparar_clusters(r_auto, r_bench, algoritmo)
            resultados.append(resultado)

            print(f"\n Resultado {algoritmo}:")
            for metrica, valor in resultado.items():
                if metrica != "algorithm":
                    print(f"{metrica}: {valor}")

        except Exception as e:
            print(f"\n Erro ao comparar {algoritmo}: {e}")

    salvar_resultado(resultados, data_str)

