from controllers.file.Consumer import process_data
import pandas as pd
from clusters import kmeans

if __name__ == "__main__":
    # print("Iniciando execução do main.py...")
    # webid = "https://192.168.0.111:3000/Joao/profile/card#me"
    # sensorType = ["AirThermometer", "ECG", "BodyThermometer", "Glucometer", "HumiditySensor"]
    
    # process_data(webid, sensorType)  # Chama a função definida no consumer.py
    
    # print("Execução concluída.")
    
    # Lista com os nomes das colunas
    collumns = ['date', 'time', 'epoch', 'moteid', 'temperature', 'humidity', 'light', 'voltage']

    # Carregar dados
    df = pd.read_csv("../data.txt", sep=" ", engine='python', names=collumns)
    df_1000 = df.iloc[:100000].copy()
    
    # Remove Collumns
    df2 = df_1000.drop(columns=['date','time', 'epoch', 'moteid',])
    
    X_scaled = kmeans.preprocess(df)
    optimal_k = kmeans.elbow(X_scaled)
    print (optimal_k)
