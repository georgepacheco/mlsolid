// Importing the required libraries
import * as rml from 'rocketrml';
import * as fs from 'fs';
import { RML_OPTIONS } from './Config';

export const mapper = async (msg: string, rml_path: string) => {

    // Global variable to load mapping file is
    let rml_file: any;
    
    // Load the rml file into memory (once only)
    if (!rml_file) {
        rml_file = await load_file_to_string(rml_path)
            .catch((err) => {
                console.error(err);
            });
    }
    // Map the jsonstring to RDF format, on the condition it is loaded correctly.
    if (!!rml_file) {
        var rdf_file = await rml.parseFileLive(rml_file.toString(), { input: msg }, RML_OPTIONS).catch((err) => {
            console.error(err);
        });

        return rdf_file;
    }
}

// Function to load file contents to a string
function load_file_to_string(filename: string) {
    return new Promise((resolve, reject) => {
        fs.readFile(filename, 'utf8', (err, data) => {
            if (err) {
                reject(err);
            } else {
                resolve(data.toString());
            }
        });
    });
}