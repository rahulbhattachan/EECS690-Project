import { ocr } from 'llama-ocr';
import dotenv from 'dotenv';

// Load environment variables from .env file
dotenv.config();

// Get the file path from command-line arguments
const filepath = process.argv[2]; // Command-line arguments are in process.argv; the third element is the first user argument

if (!filepath) {
    console.error("Error: No file path provided. Usage: node program.js <file_path>");
    process.exit(1); // Exit the program with an error code
}

// Perform OCR using the provided file path and API key
const runOCR = async () => {
    try {
        const markdown = await ocr({
            filePath: filepath,
            apiKey: process.env.TOGETHER_API_KEY,
        });
        console.log(markdown);
    } catch (error) {
        console.error("Error during OCR processing:", error);
    }
};

// Run the OCR function
runOCR();

