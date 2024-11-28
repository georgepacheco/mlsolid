import { Router } from "express";
import { StatusCodes } from 'http-status-codes';
import { login } from "../shared/middlewares";
import { SolidController } from "../controllers";

const router = Router();

router.get('/', (req, res) => {
    return res.send('Hello API Machine Learning for CSS!');
});

// router.get('/sensor/:type/observations', SolidController.getObservationsBySensorType);

// router.get('/getFot', (req, res) => {
//     return res.send('Get Fot Test');
// });

// // router.post('/login', loginBodyValidation, loginExternal);

// router.get('/sensors', SolidController.allSensorsValidation, SolidController.getAllSensors);

// router.get('/sensor/:id/observations', SolidController.observationQueryValidation,
//     SolidController.observationBodyValidation,
//     SolidController.getObservationsBySensor);

// router.get('/getAuthorization', SolidController.userValidation, SolidController.getAuthorizationToken);

// router.post('/profile', SolidController.getProfile);

// router.post ('/save', SolidController.saveValidation, SolidController.save);

// router.post('/savefot', SolidController.saveFromFot);

// router.post ('/sender', SolidController.sender);

// router.post('/remove', SolidController.remove);

// router.post('/teste', SolidController.teste);

export { router };