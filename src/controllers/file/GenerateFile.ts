import { getObservationsBySensorType } from "../solid/GetObservationsBySensorType";

const fs = require('fs');


export const generateFile = async (webId: string, sensorType: string[]) => {

    const data = await fetchDataFromRepository(webId,sensorType[0]);
    

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


process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";

const outputPath = process.argv[2];
const webid = process.argv[3];
const sensorType = process.argv[4].split(',');

// console.log("Sensor Types: " + sensorType[0]);

// const outputPath = 'SensorData.json';

if (!outputPath) {
    console.error("Erro: Caminho do arquivo de saída não fornecido.");
    process.exit(1);
} 

generateFile (webid, sensorType);