import { Request, Response } from "express";
import { IObservation, ISensor, IUser } from "../../database/models";
import { fetch as fetch2 } from "cross-fetch";
import { StatusCodes } from "http-status-codes";
import * as yup from 'yup';
import { Bindings } from "rdflib/lib/types";
import { BindingsStream } from '@comunica/types';
import { QueryEngine } from "@comunica/query-sparql-solid";
import { login } from "../../shared/middlewares/Login";


interface IParamProps {
    type?: string;
}

/**
 * 
 * @param webId of the user's repository from which the data will be obtained. 
 * @param sensorType 
 * @returns all sensors of the type 'sensorType' and their observations
 */
export const getObservationsBySensorType = async (webId: string, sensorType: string) => {
    // export const getObservationsBySensorType = async (req: Request<IParamProps, {}, IUser>, res: Response) => {

    const authFetch = await login();


    // const sourcePath = req.body.idp + req.body.podname + "/private/store.ttl";
    const sourcePath = getSourcePath(webId) + `/private/sensors/${sensorType}.ttl`;

    const myEngine = new QueryEngine();

    let query = await queryObservationBySensor(sensorType);

    const bindingsStream = await myEngine.queryBindings(query,
        {
            sources: [sourcePath],
            fetch: authFetch,
            //destination: { type: 'patchSparqlUpdate', value: sourcePath }
        });

    const sensors = await prepareDataObservations(bindingsStream, sensorType);


    return sensors;
};


const queryObservationBySensor = (sensor: string) => {

    let query = `
        PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#>
        PREFIX iot-lite: <http://purl.oclc.org/NET/UNIS/fiware/iot-lite#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX m3: <http://purl.org/iot/vocab/m3-lite#>
        PREFIX sosa: <http://www.w3.org/ns/sosa/>
        PREFIX ssn: <https://www.w3.org/ns/ssn/>
        PREFIX map: <http://example.com/soft-iot/>

        SELECT ?sensor ?observation ?resultvalue ?resulttime
        WHERE {

            ?sensor sosa:madeObservation ?observation .
            ?observation sosa:hasSimpleResult ?resultvalue .
            ?observation sosa:resultTime ?resulttime
        }
    `
    return query;
}

const getSourcePath = (webid: string) => {

    const parts = webid.split('/'); // Divide a string em partes usando '/' como delimitador
    const baseUrl = parts.slice(0, 4).join('/'); // Seleciona as primeiras 4 partes e junta-as novamente com '/'

    return baseUrl;
}

const prepareDataObservations = async (bindingsStream: BindingsStream, sensorType: string) => {

    let sensors: ISensor[] = [];
    
    for await (const binding of bindingsStream) {
        
        let sensor: ISensor = {
            sensor: '',
            lat: '',
            long: '',
            parentClass: '',
            quantityKind: '',
            sensorType: sensorType,
            unitType:'',         
            observation: []   
        }

        sensor.sensor = binding.get('sensor')?.value;

        // let observations: IObservation[] = [];
        let obs: IObservation = {
            observationId: '',
            resultValue: '',
            resultTime: ''
        };        

        obs.observationId = binding.get('observation')?.value;
        obs.resultValue = binding.get('resultvalue')?.value;
        obs.resultTime = binding.get('resulttime')?.value;        
        
        const existingSensor = sensors.find(existing => existing.sensor === sensor.sensor);
        
        if (!existingSensor) {
            sensor.observation?.push(obs);
            sensors.push(sensor);
        } else {
            existingSensor.observation?.push(obs);
        }    
    }

    return sensors;
}

// async function queryObservationBySensor(sensor: string | undefined) {

//     let query = `
//         PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#>
//         PREFIX iot-lite: <http://purl.oclc.org/NET/UNIS/fiware/iot-lite#>
//         PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
//         PREFIX m3: <http://purl.org/iot/vocab/m3-lite#>
//         PREFIX sosa: <http://www.w3.org/ns/sosa/>
//         PREFIX ssn: <https://www.w3.org/ns/ssn/>
//         PREFIX map: <http://example.com/soft-iot/>

//         SELECT ?observation ?resultvalue ?resulttime
//         WHERE {

//             map:` + sensor + ` sosa:madeObservation ?observation .
//             ?observation sosa:hasSimpleResult ?resultvalue .
//             ?observation sosa:resultTime ?resulttime
//         }
//     `
//     return query;
// }

// async function doReturn(bindingsStream: BindingsStream) {


//     let observations: IObservation[] = [];

//     for await (const binding of bindingsStream) {
//         // console.log(binding.toString());
//         let obs: IObservation = {
//             observationId: '',
//             resultValue: '',
//             resultTime: ''
//         };

//         obs.observationId = binding.get('observation')?.value;        
//         obs.resultValue = binding.get('resultvalue')?.value;
//         obs.resultTime = binding.get('resulttime')?.value;
//         observations.push(obs);
//     }

//     return observations;
// }