import * as getAllSensors from './GetAllSensors';
import * as getObservations from './GetObservationsBySensor';


export const SolidController = {
    ...getAllSensors,
    ...getObservations,
};


