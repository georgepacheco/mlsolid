// import { URI, SENSORTYPES, UNITS_OM, MEASURES_OM, M3_SENSORTYPES } from "./Config";
import { v4 as uuidv4 } from 'uuid';

export const preprocess = async (data: any)  => {
    data.forEach(function (d: any) {
        
        // complement device uri
        // d.header.device = URI + d.header.device;

        // add unit and quantityKind
        let unit: string;        
        let quantityKind: string;
        // switch (d.header.sensor){
        //     case SENSORTYPES.HUMIDITY:
        //         unit = UNITS_OM.RH;
        //         quantityKind = MEASURES_OM.RH;
        //         d.header.sensorType = M3_SENSORTYPES.HUMIDITY;
        //         d.header.parentClass = "SensingDevice";
        //         break;                
        //     case SENSORTYPES.TEMPERATURE:
        //         unit = UNITS_OM.C;                
        //         quantityKind = MEASURES_OM.C;
        //         d.header.sensorType = M3_SENSORTYPES.TEMPERATURE;
        //         d.header.parentClass = "SensingDevice";
        //         break;
        //     case SENSORTYPES.BODY_TEMPERATURE:
        //         unit = UNITS_OM.C;                
        //         quantityKind = MEASURES_OM.C;
        //         d.header.sensorType = M3_SENSORTYPES.BODY_TEMPERATURE;
        //         d.header.parentClass = "SensingDevice";
        //         break;
        //     case SENSORTYPES.ENV_TEMPERATURE:
        //         unit = UNITS_OM.C;                
        //         quantityKind = MEASURES_OM.C;
        //         d.header.sensorType = M3_SENSORTYPES.ENV_TEMPERATURE;
        //         d.header.parentClass = "SensingDevice";
        //         break;
        //     case SENSORTYPES.SOIL:
        //         unit = UNITS_OM.RH;
        //         quantityKind = MEASURES_OM.RH;
        //         d.header.sensorType = M3_SENSORTYPES.SOIL;
        //         d.header.parentClass = "SensingDevice";
        //         break;   
        //     case SENSORTYPES.GLUCOMETER:
        //         unit = "none";
        //         quantityKind = "none";    
        //         d.header.sensorType = M3_SENSORTYPES.BLOOD_GLUCOSE;        
        //         d.header.parentClass = "SensingDevice";
        //         break;                   
        //     default:
        //         unit = "none";
        //         quantityKind = "none";
        //         d.header.sensorType = "none";
        //         break;
        // }
        // d.header.unit = unit;
        // d.header.quantityKind = quantityKind;
        
        // add uuid
        d.header.deviceUUID = uuidv4();
        d.header.observationUUID = uuidv4();
        
    });
    return data;    
}



