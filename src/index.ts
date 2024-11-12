import { httpsServer } from "./server/Server";

httpsServer.listen(process.env.PORT || 3033, () => { 
    console.log(`App running in port ${process.env.PORT || 3033}`);
});