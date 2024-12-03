import { ocr } from 'llama-ocr';
import  dotenv  from 'dotenv';
import * as readline from 'readline';

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});
const question = (questionText: string) =>
    new Promise<string>(resolve => rl.question(questionText, resolve))
        .finally(() => rl.close());
const filepath = await question('Enter the path to the image file: ');
dotenv.config();
const markdown = await ocr({
  filePath: filepath, 
  apiKey: process.env.TOGETHER_API_KEY,
});console.log(markdown);
