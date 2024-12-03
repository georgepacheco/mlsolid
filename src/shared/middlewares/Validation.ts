import { RequestHandler } from "express";
import { StatusCodes } from "http-status-codes";
import * as yup from 'yup';




type TValidation = (field: 'body' | 'header' | 'params' | 'query', schema:yup.AnyObjectSchema) => RequestHandler;

export const validation: TValidation = (field, schema) => async (req, res, next) => {
    
    try {
        // abortEarly - permite devolver msg informando a quantidade de campos com erro de validação
        await schema.validate(req[field], { abortEarly: false });
        return next();
    } catch (err) {
        const yupError = err as yup.ValidationError;
        const errors: Record<string, string> = {};

        yupError.inner.forEach(error => {
            if (!error.path) return;
            errors[error.path] = error.message
        })

        return res.status(StatusCodes.BAD_REQUEST).json({ errors });
    }
}





// type TValidation = (field: 'body' | 'header' | 'params' | 'query', schema:yup.AnyObjectSchema) => RequestHandler;

// export const validation: TValidation = (field, schema) => async (req, res, next) => {
    
//     try {
//         // abortEarly - permite devolver msg informando a quantidade de campos com erro de validação
//         await schema.validate(req[field], { abortEarly: false });
//         return next();
//     } catch (err) {
//         const yupError = err as yup.ValidationError;
//         const errors: Record<string, string> = {};

//         yupError.inner.forEach(error => {
//             if (!error.path) return;
//             errors[error.path] = error.message
//         })

//         return res.status(StatusCodes.BAD_REQUEST).json({ errors });
//     }
// }