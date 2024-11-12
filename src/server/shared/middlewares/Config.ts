
// Mapping constants (which SenML unit maps to which om-2 unit and quantity) (SSN/SAREF)
//export const RML_FILE = './src/rml/rml_test.ttl';
//export const RML_FILE = './src/rml/rml_ssn.ttl';

// export const RML_LOCAL = '../solid/dist/rml_lite.ttl';
export const RML_LOCAL = './src/server/resources/rml/rml_lite.ttl';
export const RML_CLOUD = '../solid/dist/rml_lite_cloud.ttl';
export const RML_TESTE = './src/server/museum-model.rml.ttl';

/**
 * Receive file from Python script
 */
// export const RML_FILE = '../solid/dist/server/resources/rml/rml_lite.ttl'; 

export const RML_OPTIONS = {
    toRDF: true,
    verbose: false,
    xmlPerformanceMode: false,
    replace: false
}

export const URI = "http://example.com/soft-iot/sensor/";

// SENSORS TYPES
export const SENSORTYPES = {
    HUMIDITY: "humiditySensor",
    TEMPERATURE: "temperatureSensor",
    ENV_TEMPERATURE: "environmentTemperatureSensor",
    SOIL: "soilmoistureSensor",
    HEART_RATE: "heartrateSensor",
    BLOOD_PRESSURE: "bloodPressureSensor",
    BODY_TEMPERATURE: "bodyTemperatureSensor",
    ECG: "ecgmonitor",
    GLUCOMETER: "glucometerSensor",
    OXYMETER: "oxymeterSensor",
    SMOKE: "smokeSensor",

}

export const M3_SENSORTYPES = {
    HUMIDITY: "HumiditySensor",
    TEMPERATURE: "Thermometer",
    ENV_TEMPERATURE: "AirThermometer",
    SOIL: "SoilHumiditySensor",
    BLOOD_GLUCOSE: "Glucometer",
    HEART_RATE: "HeartBeatSensor",
    BLOOD_PRESSURE: "BloodPressureSensor",
    BODY_TEMPERATURE: "BodyThermometer",
    ECG: "ECG",
    GLUCOMETER: "Glucometer",
    OXYMETER: "PulseOxymeter",
    SMOKE: "SmokeDetector",
}

/**
 * @description UNITS
 */
export const M3_UNITS = {

    /**
     * @description degree celsius
     */
    "DegreeCelsius": "DegreeCelsius",

    /**
     * @description meter
     */
    "Meter": "Meter",

    /**
     * @description Glucose unit -  Millimol per liter 
     */
    "MmolPerLiter": "MmolPerLiter",

    /**
         * @description MillimeterMercury mmHg unit is used to measure blood pressure measurements (systolic and diastolic)
         */
    "MmHg": "MmHg",

    /**
     * @description percent
     */
    "Percent": "Percent",

    "PPM": "PPM"

}

/**
 * @description QUANTITY UNIT
 */
export const M3_QU = {

    /**
     * @description Blood Glucose level, blood sugar level
     */
    "BloodGlucose": "BloodGlucose",

    /**
     * @description BloodPressure
     */
    "BloodPressure":"BloodPressure",

    /**
     * @description BodyTemperature
     */
    "BodyTemperature": "BodyTemperature",

    /**
    * @description
    */
    "Humidity": "Humidity",

    /**
     * @description The ratio of vapour pressure to saturation 
     * vapour pressure, where vapour pressure is the pressure 
     * exerted by the molecules of water vapour and saturation 
     * vapour pressure is the pressure exerted by molecules of 
     * water vapour in AIR that has attained saturation.
     */
    "RelativeHumidity": "RelativeHumidity",

    /**
     * @description Soil Humidity 
     */
    "SoilHumidity": "SoilHumidity",

    /**
         * @description SPO2 measurement
        */
    "SPO2": "SPO2",

    /**
     * @description By defaut, Air temperature
     */
    "Temperature": "Temperature",

    /**
     * @description Chemical Agent Atmospheric Concentration
     */
    "ParticleConcentration": "ChemicalAgentAtmosphericConcentration",
}

export const SERVER_ADDRESS="http://localhost:3000/"

