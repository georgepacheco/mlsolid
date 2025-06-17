import json
import pandas as pd
import subprocess
import tempfile
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from clusters import shared
from datetime import datetime

def process_data(webid, sensorType, limit, output_csv_path): 
    # Arquivo temporário para comunicação
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
        temp_file_path = temp_file.name
        
        try:
            # Converte a lista de sensores em uma string separada por vírgulas
            sensor_types_str = ",".join(sensorType)
                       
            # Executa o script Node.js para gerar os dados
            subprocess.run(["node", "../../../dist/controllers/file/GenerateFile.js", temp_file_path, webid, sensor_types_str, str(limit)], check=True)

            # Consome os dados e salva como DataFrame
            X_scaled = prepare_data(temp_file_path, output_csv_path)

        except subprocess.CalledProcessError as e:
            print("Erro ao executar o script Node.js:", e)
            X_scaled = None

        finally:
            os.remove(temp_file_path)
            print("Arquivo temporário removido.")

        return X_scaled

def prepare_data(file_path, output_csv_path):
    with open(file_path, 'r') as file:
        data = json.load(file)        

        sensor_columns = {}

        for sensor_data in data:
            sensor_type = sensor_data['sensorType']  
            if 'observation' in sensor_data:
                if sensor_type not in sensor_columns:
                    sensor_columns[sensor_type] = []
                
                for observation in sensor_data['observation']:
                    result_value = observation.get('resultValue', pd.NA)
                    sensor_columns[sensor_type].append(result_value)

        max_length = max(len(values) for values in sensor_columns.values())

        for sensor_type, values in sensor_columns.items():
            sensor_columns[sensor_type].extend([pd.NA] * (max_length - len(values)))

        df = pd.DataFrame(sensor_columns)
        print(df.info())

        # Salva o DataFrame como CSV
        df.to_csv(output_csv_path, sep=",", index=False)
        print(f"Dados salvos em: {output_csv_path}")

        # Preprocessamento (opcional)
        X_scaled = shared.preprocess(df)
        
        return X_scaled

if __name__ == "__main__":
    
    print("Iniciando execução do Consumer...")
    webid = "https://192.168.0.111:3000/Joao/profile/card#me"
    
    sensorType_env = ["HumiditySensor", "AirThermometer", "CO_Sensor", "LightSensor"]
    
    data_atual = datetime.now().strftime("%Y%m%d")
    
    output = os.path.join("dataset", f"dataset_{data_atual}.csv")
    
    qtd = 1000 
    # nao estou mais usando o qtd
    process_data(webid, sensorType_env, qtd, output)
    

