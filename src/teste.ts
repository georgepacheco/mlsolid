import { SolidController } from "./server/controllers";
import { IUser } from "./server/database/models";
import { login } from "./server/shared/middlewares";


const doService =  async () => {

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

    const result = await SolidController.getObservationsBySensorType(user, sensorType);
    // console.log(result);

    // const authFetch = await login();

    // console.log(authFetch);
}

doService();