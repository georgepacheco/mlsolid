import { Request, Response } from "express";
import { IObservation, IUser } from "../../database/models";
import { login, validation } from "../../shared/middlewares";
// import fetch from 'node-fetch';
import { fetch as fetch2 } from "cross-fetch";
import { StatusCodes } from "http-status-codes";
import * as yup from 'yup';
import { Bindings } from "rdflib/lib/types";
import { BindingsStream } from '@comunica/types';
import { QueryEngine } from "@comunica/query-sparql-solid";


interface IParamProps {
    id?: string;
}

const bodyValidation: yup.ObjectSchema<IUser> = yup.object().shape({
    userid: yup.string().required(),
    local_webid: yup.string().required(),
    webid: yup.string().required(),
    idp: yup.string().required(),
    username: yup.string().required(),
    password: yup.string().required(),
    podname: yup.string().required()
});

const queryValidation: yup.ObjectSchema<IParamProps> = yup.object().shape({
    id: yup.string().required(),
});

export const observationBodyValidation = validation('body', bodyValidation);
export const observationQueryValidation = validation('params', queryValidation);

export const getObservationsBySensor = async (req: Request<IParamProps, {}, IUser>, res: Response) => {

    const authFetch = await login(req, res);

    // console.log("Teste .ENV: " + process.env.SOLID_IDP);
    const sourcePath = req.body.idp + req.body.podname + "/private/store.ttl";

    const myEngine = new QueryEngine();

    let query = await queryObservationBySensor(req.params.id?.toString());

    const bindingsStream = await myEngine.queryBindings(query,
        {
            sources: [sourcePath],
            fetch : authFetch,
            //destination: { type: 'patchSparqlUpdate', value: sourcePath }
        });

    const observations = await doReturn(bindingsStream);

    return res.status(StatusCodes.OK).json(observations);

    // return res.send('olá');
};


async function queryObservationBySensor(sensor: string | undefined) {

    let query = `
        PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#>
        PREFIX iot-lite: <http://purl.oclc.org/NET/UNIS/fiware/iot-lite#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX m3: <http://purl.org/iot/vocab/m3-lite#>
        PREFIX sosa: <http://www.w3.org/ns/sosa/>
        PREFIX ssn: <https://www.w3.org/ns/ssn/>
        PREFIX map: <http://example.com/soft-iot/>

        SELECT ?observation ?resultvalue ?resulttime
        WHERE {

            map:` + sensor + ` sosa:madeObservation ?observation .
            ?observation sosa:hasSimpleResult ?resultvalue .
            ?observation sosa:resultTime ?resulttime
        }
    `
    return query;
}

async function doReturn(bindingsStream: BindingsStream) {


    let observations: IObservation[] = [];

    for await (const binding of bindingsStream) {
        // console.log(binding.toString());
        let obs: IObservation = {
            observationId: '',
            resultValue: '',
            resultTime: ''
        };

        obs.observationId = binding.get('observation')?.value;        
        obs.resultValue = binding.get('resultvalue')?.value;
        obs.resultTime = binding.get('resulttime')?.value;
        observations.push(obs);
    }

    return observations;
}