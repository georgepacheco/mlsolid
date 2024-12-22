import { login } from "../../shared/middlewares/Login";
import { getFile, getSourceUrl, overwriteFile } from "@inrupt/solid-client";

export interface IMetrics  {
    silhouette: number,
    davies_bouldin: number
    calinski_harabasz: number
}

const getSourcePath = (webid: string) => {

    const parts = webid.split('/'); // Divide a string em partes usando '/' como delimitador
    const baseUrl = parts.slice(0, 4).join('/'); // Seleciona as primeiras 4 partes e junta-as novamente com '/'

    return baseUrl;
}

export const saveMetrics = async (webId: string, metrics: IMetrics) => {
    
    const authFetch = await login();

    const sourcePath = getSourcePath(webId) + `/private/metrics.json`;        

    try {
        await overwriteFile(
            sourcePath,
            new File([JSON.stringify(metrics)], 'metrics.json', { type: "application/json" }),
            { fetch: authFetch }
        );
    } catch (error) {
        return new Error((error as { message: string }).message || 'Error saving the metrics.');
    }
}

export const getMetrics = async (webId: string): Promise<IMetrics | Error> => {
    
    const authFetch = await login();
    const sourcePath = getSourcePath(webId) + `/private/metrics.json`;        
    try {
        const fileBlob = await getFile(sourcePath, { fetch: authFetch });
        const text = new TextDecoder().decode(await fileBlob.arrayBuffer());
        const data: IMetrics = JSON.parse(text);
        let metrics: IMetrics = {
            silhouette: data.silhouette,            
            davies_bouldin: data.davies_bouldin,
            calinski_harabasz: data.calinski_harabasz,
        };
        return metrics;
    } catch (error) {
        return new Error((error as { message: string }).message || 'Error saving cloud list file.');     
    }
    
}
