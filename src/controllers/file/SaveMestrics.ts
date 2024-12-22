import { IMetrics, saveMetrics } from "../solid/Metrics";

process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";

const webId = process.argv[2];
const data = process.argv[3]

const [silhouette, davies_bouldin, calinski_harabasz] = data
    .split(',')
    .map(value => parseFloat(value.trim()));

const metrics: IMetrics = {
    silhouette,
    davies_bouldin,
    calinski_harabasz
};
    
console.log(metrics);
saveMetrics(webId, metrics);