from controllers.file.Consumer import process_data
import pandas as pd
from clusters import kmeans

if __name__ == "__main__":
    # print("Iniciando execução do main.py...")
    # webid = "https://192.168.0.111:3000/Joao/profile/card#me"
    # sensorType = ["AirThermometer", "HumiditySensor"]
    
    # process_data(webid, sensorType)  # Chama a função definida no consumer.py
    
    # print("Execução concluída.")
    

    # DATASET INTELAB ======================================================================
    # Lista com os nomes das colunas
    # collumns = ['date', 'time', 'epoch', 'moteid', 'temperature', 'humidity', 'light', 'voltage']

    # # Carregar dados
    # df = pd.read_csv("../data.txt", sep=" ", engine='python', names=collumns)
    # df_1000 = df.iloc[:10000].copy()
    
    # # Remove Collumns
    # df2 = df_1000.drop(columns=['date','time', 'epoch', 'moteid',])
    

    # DATASET SAÚDE =========================================================================
    # collumns = ['Age','Blood Glucose Level(BGL)','Diastolic Blood Pressure','Systolic Blood Pressure','Heart Rate','Body Temperature','SPO2','Sweating  (Y/N)','Shivering (Y/N)','Diabetic/NonDiabetic (D/N)']
    
    # # Carregar dados
    # df = pd.read_csv("data/pt9.csv", sep=",", engine='python', names=collumns)
    
    # # Remove Collumns
    # df2 = df.drop(columns=['Diabetic/NonDiabetic (D/N)'])
    
    # df2 = df2.apply(pd.to_numeric, errors='coerce')
    # # print (df2.info())

    # DATASET CONJUNTO =========================================================================        
    # collumns = ['Age','Blood Glucose Level(BGL)','Diastolic Blood Pressure','Systolic Blood Pressure','Heart Rate','Body Temperature','SPO2','Sweating  (Y/N)','Shivering (Y/N)','Diabetic/NonDiabetic (D/N)']
    
    # Carregar dados
    df = pd.read_csv("data/health_intel.csv", sep=",", engine='python')
    
    # Remove Collumns
    df2 = df.drop(columns=['Age', 'Diabetic/NonDiabetic (D/N)'])
    
    df2 = df2.apply(pd.to_numeric, errors='coerce')
    # print (df2.info())
    
   
    # PROCESSAMENTO =========================================================================
    X_scaled = kmeans.preprocess(df2)
    # print (X_scaled)
    
    optimal_k = kmeans.elbow(X_scaled)
    
    results, params = kmeans.run_kmeans(X_scaled, optimal_k)
    
    print ("Clusters: ", optimal_k)
    print("Melhores Resultados (Silhouete, Davies_Bouldin, Calinski_Harabasz):", results)
