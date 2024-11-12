import express from 'express';
import cors from 'cors';
import {router} from './routes';
import 'dotenv/config';
import https from 'https';
import fs from 'fs';

// Caminho para o certificado e chave privada no docker
// const privateKey = fs.readFileSync('../solid/server.key', 'utf8');
// const certificate = fs.readFileSync('../solid/server.cert', 'utf8');
// const credentials = { key: privateKey, cert: certificate };

// Caminho para o certificado e chave privada LOCAL
const privateKey = fs.readFileSync('../mlsolid/certificates/server.key', 'utf8');
const certificate = fs.readFileSync('../mlsolid/certificates/server.cert', 'utf8');
const credentials = { key: privateKey, cert: certificate };

const server = express();

// server.use(cors({
//     // origin: '*'
//     origin: process.env.ENABLED_CORS?.split(';') || []
// }));

const basePath = '/apiCloud';

server.use(cors());

server.use(basePath, express.json());

server.use(basePath, router);

// Criação do servidor HTTPS
const httpsServer = https.createServer(credentials, server);


export { httpsServer };

// export { server };