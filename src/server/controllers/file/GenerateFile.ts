import { getObservationsBySensorType } from "../solid/GetObservationsBySensorType";

const fs = require('fs');

// Lê o caminho do arquivo de saída como argumento
const outputPath = 'SensorData.json';

export const generateFile = async (webId: string, sensorType: string) => {

    if (!outputPath) {
        console.error("Erro: Caminho do arquivo de saída não fornecido.");
        process.exit(1);
    }

    const data = await fetchDataFromRepository(webId,sensorType);
    

    // Escreve os dados no arquivo JSON
    fs.writeFileSync(outputPath, JSON.stringify(data, null, 2));
    console.log("Dados escritos no arquivo:", outputPath);
}

// Simula a obtenção de dados de um repositório
const fetchDataFromRepository = async (webId: string, sensorType: string) => {
    const result = await getObservationsBySensorType(webId, sensorType);
    console.log(result);
    return result;
}
