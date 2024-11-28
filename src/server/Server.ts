import express from 'express';
import cors from 'cors';
import {router} from './routes';
import 'dotenv/config';
import https from 'https';
import fs from 'fs';
import path from 'path';

// Caminho para o certificado e chave privada no docker
// const privateKey = fs.readFileSync('../solid/server.key', 'utf8');
// const certificate = fs.readFileSync('../solid/server.cert', 'utf8');
// const credentials = { key: privateKey, cert: certificate };


const server = express();

// server.use(cors({
//     // origin: '*'
//     origin: process.env.ENABLED_CORS?.split(';') || []
// }));

const basePath = '/apimlsolid';

server.use(cors());

server.use(basePath, express.json());

server.use(basePath, router);

// Criação do servidor HTTPS
// const httpsServer = https.createServer(credentials, server);

// export { httpsServer };

export { server };