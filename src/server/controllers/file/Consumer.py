import json
import tempfile
import subprocess

def prepare_data(file_path):
    # Lê os dados do arquivo
    with open(file_path, 'r') as file:
        data = json.load(file)
        print("Dados consumidos do arquivo:", data)

        # Prepara os dados para serem usados pelos algoritmos
        # Criar uma tabela?    
        
        # # Processa os dados
        # for item in data['data']:
        #     print(f"Processando ID: {item['id']}, Valor: {item['value']}")

def process_data (webid, sensorType): 
    # Arquivo temporário para comunicação
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
        temp_file_path = temp_file.name
                
        try:
            # Executa o script Node.js para gerar os dados
            subprocess.run(["node", "../dist/server/controllers/file/GenerateFile.js", temp_file_path, webid, sensorType], check=True)

            # Consome os dados gerados pelo Node.js
            prepare_data(temp_file_path)

        except subprocess.CalledProcessError as e:
            print("Erro ao executar o script Node.js:", e)

        finally:
            # Remove o arquivo temporário, se necessário
            import os
            os.remove(temp_file_path)
            print("Arquivo temporário removido.")
