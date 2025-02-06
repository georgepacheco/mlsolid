import time
import psutil
import os
from typing import Callable, Any

def medir_performance(algoritmo: Callable, *args, **kwargs) -> dict:
    num_nucleos = psutil.cpu_count(logical=True)  # Obtém a quantidade de núcleos
    processo = psutil.Process(os.getpid())  # Obtém o processo atual

    # Capturar o uso total de CPU antes da execução
    cpu_inicio = processo.cpu_times()
    memoria_inicio = processo.memory_info().rss
    inicio = time.time()

    # Executa o algoritmo
    resultado = algoritmo(*args, **kwargs)

    # Capturar o uso total de CPU depois da execução
    cpu_fim = processo.cpu_times()
    memoria_fim = processo.memory_info().rss
    fim = time.time()

    # Calcular o tempo total de CPU usado pelo processo
    tempo_cpu_total = (cpu_fim.user - cpu_inicio.user) + (cpu_fim.system - cpu_inicio.system)

    # Normalizar o uso de CPU pelo número de núcleos
    uso_cpu_real = (tempo_cpu_total / (fim - inicio)) * 100 / num_nucleos

    tempo_execucao = fim - inicio
    uso_memoria = (memoria_fim - memoria_inicio) / (1024 * 1024)  # Convertendo para MB

    return {
        "tempo_execucao": tempo_execucao,
        "uso_memoria_MB": uso_memoria,
        "uso_cpu_percent": uso_cpu_real,
        "resultado": resultado
    }
