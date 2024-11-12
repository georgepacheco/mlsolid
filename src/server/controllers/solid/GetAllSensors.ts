import { Request, Response } from "express";
import { QueryEngine } from '@comunica/query-sparql-solid';
import { login, validation } from "../../shared/middlewares";
import { ISensor, IUser } from "../../database/models";
// import fetch from 'node-fetch';
import { fetch as fetch2 } from "cross-fetch";
import { StatusCodes } from "http-status-codes";
import * as yup from 'yup';
import { Bindings } from "rdflib/lib/types";
import { BindingsStream } from '@comunica/types';


const bodyValidation: yup.ObjectSchema<IUser> = yup.object().shape({
    userid: yup.string().required(),
    local_webid: yup.string().required(),
    webid: yup.string().required(),
    idp: yup.string().required(),
    username: yup.string().required(),
    password: yup.string().required(),
    podname: yup.string().required()
});

export const allSensorsValidation = validation('body', bodyValidation);

export const getAllSensors = async (req: Request<{}, {}, IUser>, res: Response) => {

    const authFetch = await login(req, res);

    // const sourcePath = process.env.SOLID_IDP + req.body.podname + "/private/store.ttl";
    const sourcePath = req.body.idp + req.body.podname + "/private/store.ttl";
    
    const myEngine = new QueryEngine();

    let query = await querySelectSensorByUser();

    const bindingsStream = await myEngine.queryBindings(query,
        {
            sources: [sourcePath],
            fetch: authFetch,
            //destination: { type: 'patchSparqlUpdate', value: sourcePath }
        });

    const sensors = await doReturn(bindingsStream);

    return res.status(StatusCodes.OK).json(sensors);
}

async function querySelectSensorByUser() {
    let query = `
        PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#>
        PREFIX iot-lite: <http://purl.oclc.org/NET/UNIS/fiware/iot-lite#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX m3: <http://purl.org/iot/vocab/m3-lite#>
        PREFIX sosa: <http://www.w3.org/ns/sosa/>
        PREFIX ssn: <https://www.w3.org/ns/ssn/>
        
        SELECT ?sensor ?parentClass ?sensorType ?coverage ?point ?latitude ?longitude ?unitType ?quKind
        WHERE {
            ?sensor rdfs:subClassOf ssn:SensingDevice .
            ?sensor rdfs:subClassOf ?parentClass .
            ?sensor rdf:type ?sensorType .
            ?sensor iot-lite:hasCoverage ?coverage .
            ?coverage geo:location ?point .
            ?point geo:lat ?latitude .
            ?point geo:long ?longitude .
            ?sensor iot-lite:hasUnit ?unitType .
            ?sensor iot-lite:quantityKind ?quKind
        }   
    
    `;

    return query;
}

async function doReturn(bindingsStream: BindingsStream) {


    let sensors: ISensor[] = [];

    for await (const binding of bindingsStream) {
        // console.log(binding.toString());
        let sensor: ISensor = {
            sensor: '',
            sensorType: '',
            lat: '',
            long: '',            
            unitType: '',
            quantityKind: '',
            parentClass: ''
        };

        sensor.sensor = binding.get('sensor')?.value;
        sensor.sensorType = binding.get('sensorType')?.value;
        sensor.lat = binding.get('latitude')?.value;
        sensor.long = binding.get('longitude')?.value;
        sensor.unitType = binding.get('unitType')?.value;           
        sensor.quantityKind = binding.get('quKind')?.value;           
        sensor.parentClass = binding.get('parentClass')?.value;
        sensors.push(sensor);
    }

    return sensors;
}