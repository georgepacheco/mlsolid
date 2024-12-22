import time
from memory_profiler import memory_usage

def profile_memory_and_time(func):
    def wrapper(*args, **kwargs):
        # Medir uso de memória inicial
        mem_before = memory_usage()[0]

        # Medir tempo inicial
        start_time = time.time()

        # Monitorar o uso de memória enquanto a função é executada
        mem_during = memory_usage((func, args, kwargs), interval=0.1)

        # Medir tempo final
        end_time = time.time()
        # Medir uso de memória final
        mem_after = memory_usage()[0]

        # Calcular tempo e memória
        elapsed_time = end_time - start_time
        max_mem_during = max(mem_during) if mem_during else mem_before
        mem_used = max(max_mem_during, mem_after) - mem_before

        print(f"Função '{func.__name__}' levou {elapsed_time:.4f} segundos")
        print(f"Memória antes: {mem_before:.2f} MiB")
        print(f"Memória máxima durante: {max_mem_during:.2f} MiB")
        print(f"Memória após: {mem_after:.2f} MiB")
        print(f"Uso total de memória: {mem_used:.2f} MiB")

        return func(*args, **kwargs)
    return wrapper

@profile_memory_and_time
def some_function():
    data = [i ** 2 for i in range(10**6)]  # Exemplo que consome memória
    return sum(data)

if __name__ == "__main__":
    some_function()
