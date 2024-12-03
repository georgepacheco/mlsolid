from server.controllers.file.Consumer import process_data

if __name__ == "__main__":
    print("Iniciando execução do main.py...")
    webid = "https://192.168.0.111:3000/Joao/profile/card#me"
    sensorType = "AirThermometer"
    
    process_data(webid, sensorType)  # Chama a função definida no consumer.py
    
    print("Execução concluída.")