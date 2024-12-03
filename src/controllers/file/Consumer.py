import json
import tempfile
import subprocess
import pandas as pd

def prepare_data(file_path):
    # Lê os dados do arquivo
    with open(file_path, 'r') as file:
        data = json.load(file)

        # Prepara os dados para serem usados pelos algoritmos        
        # sc01_date - sc01_value
        # Lista para armazenar os dados extraídos
        table_data = []
        
        # # Processa os dados
        for sensor_data in data:
            sensor_name = sensor_data['sensor']            
            if 'observation' in sensor_data:
                for observation in sensor_data['observation']:
                    result_value = observation.get('resultValue', 'N/A')
                    result_time = observation.get('resultTime', 'N/A')
                    # Adiciona os dados à lista
                    table_data.append({'{sensor_name}_date': result_time, '{sensor_name}_value': result_value})
         
        # Criação da tabela com pandas
        df = pd.DataFrame(table_data)
        print(df)

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
