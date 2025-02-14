import Summary



if __name__ == "__main__":
    file_name = "../statistic_results.json"
    data = Summary.load_file(file_name)
    # Summary.time_memory_graph(data=data)
    # Summary.algo_time_memory_graph(data=data)
    # Summary.metrics_graph(data)
    Summary.metrics_domain_table(data)