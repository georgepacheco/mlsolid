import { ISensor } from "../../database/models";
import { getObservationsBySensorType } from "../solid/GetObservationsBySensorType";

const fs = require('fs');

/**
 * @description generate file with data from Solid based on user's webId and sensors types.
 * @param webId user's webId
 * @param sensorType sensors types used to build the data frame
 */
export const generateFile = async (webId: string, sensorType: string[], limit: number) => {

    const data: ISensor[] = []
    for (let i = 0; i < sensorType.length; i++) {
        const result = await fetchDataFromRepository(webId, sensorType[i], limit)
        data.push(...result);        
    }
    
    // Escreve os dados no arquivo JSON
    fs.writeFileSync(outputPath, JSON.stringify(data, null, 2));
    // console.log("Dados escritos no arquivo:", outputPath);
}


const fetchDataFromRepository = async (webId: string, sensorType: string, limit: number) => {
    const result = await getObservationsBySensorType(webId, sensorType, limit);
    // console.log(result);
    return result;
}


process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";

const outputPath = process.argv[2];
const webid = process.argv[3];
const sensorType = process.argv[4].split(',');
const limit = Number(process.argv[5]);

if (!outputPath) {
    console.error("Erro: Caminho do arquivo de saída não fornecido.");
    process.exit(1);
}

generateFile(webid, sensorType, limit);