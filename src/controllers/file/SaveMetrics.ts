import { IMetrics, saveMetrics } from "../solid/Metrics";

process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";

const webId = process.argv[2];
const data = process.argv[3];
const domain = process.argv[4];

// Remove the parentheses before performing the split
const metricsTuple = data.replace(/[()]/g, '').split(',').map(Number) as [number, number, number, string];

const metrics: IMetrics = {
    silhouette: metricsTuple[0],
    davies_bouldin: metricsTuple[1],
    calinski_harabasz: metricsTuple[2],
    domain: domain
};	
    
//console.log(metrics);
saveMetrics(webId, metrics);
