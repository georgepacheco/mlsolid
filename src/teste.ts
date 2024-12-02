import { SolidController } from "./server/controllers";
import { generateFile } from "./server/controllers/file/GenerateFile";
import { ISensor, IUser } from "./server/database/models";
import { login } from "./server/shared/middlewares";


const doService = async () => {

    process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';

    let user: IUser = {
        userid: '',
        local_webid: '',
        idp: '',
        username: '',
        password: '',
        podname: '',
        webid: "https://192.168.0.111:3000/Joao/profile/card#me"

    }

    const sensorType = 'AirThermometer';

    console.log('teste');

    // const result = await SolidController.getObservationsBySensorType(user.webid, sensorType);

    const result = await generateFile(user.webid, sensorType);

    console.log('FIM')

}

doService();