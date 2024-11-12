interface ILocation {
    lat?: number;
    long?: number;
};

interface ITime {
    collect?: number;
    publish?: number;
};

interface IHeader {
    sensor?: string;
    device?: string;
    time?: ITime;
    location?: ILocation;
}

export interface IData {
    code?: string;
    method?: string;
    header?: IHeader;
    data?: Array<string>;
    date?: string;
};

interface IObservation {
    observationId?: string;
    resultTime?: string;
    resultValue?: string;
}

export interface IDataFot {
    sensor: string;
    sensorName: string;
    sensorType: string;
    lat: string;
    long: string;
    unityType: string;
    quantityKind: string;
    parentClass: string;       
    observation: IObservation[];
}