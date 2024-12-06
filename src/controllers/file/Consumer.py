import json
import tempfile
import subprocess
import pandas as pd
from clusters import kmeans

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

        # print(df)
        
        # Chamar os algoritmo de agrupamento
        # run_kmeans(df)
        X_scaled = kmeans.preprocess(df)
        optimal_k = kmeans.elbow(X_scaled)
        print ("k =  ", optimal_k)
        kmeans.run_kmeans(X_scaled, optimal_k)
        
        # Enviar para o solid o resultado do agrupamento
        
        

def process_data (webid, sensorType): 
    # Arquivo temporário para comunicação
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
        temp_file_path = temp_file.name
                
        try:
            # Converte a lista de sensores em uma string separada por vírgulas
            sensor_types_str = ",".join(sensorType)
                       
            # Executa o script Node.js para gerar os dados
            subprocess.run(["node", "../dist/controllers/file/GenerateFile.js", temp_file_path, webid, sensor_types_str], check=True)

            # Consome os dados gerados pelo Node.js
            prepare_data(temp_file_path)

        except subprocess.CalledProcessError as e:
            print("Erro ao executar o script Node.js:", e)

        finally:
            # Remove o arquivo temporário, se necessário
            import os
            os.remove(temp_file_path)
            print("Arquivo temporário removido.")
