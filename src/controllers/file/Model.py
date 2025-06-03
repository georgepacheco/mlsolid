from typing import List, Optional
import json
import os
from dataclasses import asdict, dataclass, field
import numpy as np

@dataclass
class Algorithms:
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
class Statistics:
    qtd_data: Optional[int] = None
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
    best_algorithm: Optional[str] = None
    

@dataclass
class Domain:
    name: Optional[str] = None
    statistics: List[Statistics] = field(default_factory=list)

@dataclass
class Results:
    domains: List[Domain] = field(default_factory=list)

def convert_types(obj):
    """Converte int64, int32, float64, e outros tipos NumPy para tipos Python compatíveis com JSON."""
    if isinstance(obj, (np.integer, np.int32, np.int64)):  # Converte np.int para int
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):  # Converte np.float para float
        return float(obj)
    elif isinstance(obj, list):  # Converte listas recursivamente
        return [convert_types(item) for item in obj]
    elif isinstance(obj, dict):  # Converte dicionários recursivamente, garantindo que as chaves sejam str
        return {str(key): convert_types(value) for key, value in obj.items()}
    return obj  # Retorna o objeto se não precisar de conversão

# Classe para manipulação do JSON
class FileManager:
    def __init__(self, file_path: str = "results.json"):
        self.file_path = file_path

    def load_results(self) -> Results:
        """Carrega os dados do arquivo JSON e converte para objetos Results."""
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Convertendo dicionários para objetos das classes
                return Results(
                    domains=[
                        Domain(
                            name=d["name"],
                            statistics=[
                                Statistics(**stat) for stat in d.get("statistics", [])
                            ]
                        ) for d in data.get("domains", [])
                    ]
                )
        return Results()

    # def load_results(self) -> Results:
    #     """Carrega os dados do arquivo JSON."""
    #     if os.path.exists(self.file_path):
    #         with open(self.file_path, "r", encoding="utf-8") as f:
    #             data = json.load(f)
    #             return Results(**data)
    #     return Results()
    
    def save_results(self, results: Results):
        """Salva os dados no arquivo JSON."""
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(convert_types(asdict(results)), f, indent=4, ensure_ascii=False)

    def add_or_update_statistics(self, results: Results, domain_name: str, new_stat: Statistics):
        """Adiciona ou atualiza uma estatística no domínio correto."""
        # Verifica se o domínio já existe
        domain = next((d for d in results.domains if d.name == domain_name), None)
        
        if not domain:
            # Se o domínio não existir, cria um novo e adiciona a estatística
            domain = Domain(name=domain_name, statistics=[new_stat])
            results.domains.append(domain)
            print(f"Novo domínio '{domain_name}' adicionado.")
        else:
            # Verifica se já existe uma estatística com o mesmo qtd_data
            stat = next((s for s in domain.statistics if s.qtd_data == new_stat.qtd_data), None)
            
            if stat:
                # Atualiza os valores da estatística existente
                print(f"Atualizando estatística com qtd_data={new_stat.qtd_data} no domínio '{domain_name}'.")
                stat.__dict__.update(new_stat.__dict__)
            else:
                # Adiciona a nova estatística
                print(f"Adicionando nova estatística ao domínio '{domain_name}'.")
                domain.statistics.append(new_stat)

        # Salva os dados atualizados
        self.save_results(results)
        print("Arquivo atualizado com sucesso!")
