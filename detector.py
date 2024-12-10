## command line tool to execute program
import os
import sys
from BoundingBox import BoundingBox
from PackageExtractor import PackageExtractor
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
#from TextRecognition import text_recognition
from TextRecognition_Ollama import text_recognition
from time import time

# default paremeters
default_model = "./783-Pin-Detection/runs/detect/train56/weights/best.pt"

def help():
    print("python3 detector.py <filename> <args>\n"
          "  Available arguments:\n"
          "    -save                          : Saves both output image and text; default filenames are output.png and output.txt\n"
          "    -ifile <filename>              : Saves the output image to the filename given. -save option must be called\n"
          "    -tfile <filename>              : Saves the text output in plaintext to the filename given\n"
          "    -show                          : Shows the output as a pyplot\n"
          "    -model <model>                 : Runs a different model to the default\n"
          "    -no-text                       : Returns no text\n"
          "    -no-image                      : Returns no image\n"
          "    -overlay-good-pin <color code> : Overlays good pins with some color\n"
          "    -overlay-bent-pin <color code> : Overlays bent pins with some color\n"
          "    -overlay-package  <color code> : Overlays the package with some color\n"
          "    -overlay-text     <color code> : Overlays text with some color\n"
          "    -help                          : Brings up the help menu (this menu)\n\n"
          "\n"
          "  Color codes: c1, c2, c3, c4, c5, c6\n\n"
          "  Example:\n"
          "    The following example displays the bounding box overlayed on the image.\n"
          "    It does not save the example.\n\n"
          "        python3 detector.py image.png\n\n"
          "  Example:\n"
          "    The following example displays the bounding box overlayed on the image.\n"
          "    It does not save the example.\n\n"
          "        python3 detector.py image.png -show\n\n"
          "  Example:\n"
          "    The following example saves the image, no text, to output.png.\n\n"
          "        python3 detector.py image.png -save -no-text\n\n"
          "  Example:\n"
          "    The following example shows the image and saves the text file to some filename, no image.\n\n"
          "        python3 detector.py image.png -show -save -no-image -tfile somefilename.txt\n"
          )

class Commands:
    def __init__(self, cmd : str, inputs : int):
        self.cmd = cmd
        self.inputs = inputs

colors = {
    'c0' : [255, 0  , 0  ],
    'c1' : [0  , 255, 0  ],
    'c2' : [0  , 0  , 255],
    'c3' : [255, 255, 0  ],
    'c4' : [255, 0  , 255],
    'c5' : [0  , 255, 255],
    'c6' : [255, 255, 255]
}

commands = [
    Commands("-save", 0),
    Commands("-ifile", 1),
    Commands("-tfile", 1),
    Commands("-show", 0),
    Commands("-model", 1),
    Commands("-help", 0),
    Commands("-no-text", 0),
    Commands("-no-image", 0),
    Commands("-overlay-good-pin", 1),
    Commands("-overlay-bent-pin", 1),
    Commands("-overlay-package", 1),
    Commands("-overlay-text", 1),
    Commands("-apple-ocr", 0),
    Commands("-chat-gpt", 0),
    Commands("-llama-vision", 0)
]

def is_a_command(txt)->bool:
    for cmd in commands:
        if txt == cmd.cmd:
            return True
    return False

def num_parameters(txt)->bool:
    for cmd in commands:
        if txt == cmd.cmd:
            return cmd.inputs
    return 0

class Detector:
    def __init__(self)->None:
        self.command_tree = {}

        self.active_commands = {
            '-model'    : default_model,
            '-show'     : False,
            '-save'     : False,
            '-ifile'    :  "output.png",
            '-tfile'    : "output.txt",
            '-no-image' : False,
            '-no-text'  : False,
            '-overlay-good-pin' : None,
            '-overlay-bent-pin' : 'c0',
            '-overlay-package'  : 'c5',
            '-overlay-text'     : None,
            '-apple-ocr'        : False,
            '-chat-gpt'         : False,
            '-llama-vision'     : False
        }

    def runa_ocr(self, image : Image.Image, ocr_mode : int = 0)->list:
        import cv2
        from skimage.measure import block_reduce
        import base64
        from io import BytesIO

        if self.active_commands['-apple-ocr']:
            from apple_ocr.ocr import OCR as AppleOCR
        class AppleTextVisualization:
            def __init__(self)->None:
                pass

            @staticmethod
            def Image_To_Text(img : str | np.ndarray, minimum_horizontal_size : int = 1600, rounds : int = 4, kernel_size : int | float = 64):
                if isinstance(img, str):
                    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                if isinstance(img, Image.Image):
                    img = np.array(img.convert('L'))

                if img.shape[0] < minimum_horizontal_size:
                    ratio = minimum_horizontal_size/img.shape[0]
                    img   = cv2.resize(img.astype(np.float32), dsize=(int(img.shape[1]*ratio), minimum_horizontal_size)).astype(np.uint8)

                ii = img
                iii = AppleTextVisualization.enhance_text_for_ocr(ii, rounds)

                ocr_instance = AppleOCR(image=Image.fromarray(iii))
                try:
                    dataframe    = ocr_instance.recognize()
                except:
                    dataframe    = None
                return dataframe, iii

            @staticmethod
            def isolate_ic(img, color : str = "black", kernel_size : int | tuple = 128):
                match color:
                    case "black":
                        i = img/255<0.5
                    case "white":
                        i = img/255>0.5
                    case _:
                        raise ValueError(f"color: {color} is not an option. Choose between white or black")
                # bounding box is defined as (x, y, height, width)
                i = i.astype(np.float32)# +0.001

                # apply gaussian blur
                blur = cv2.GaussianBlur(i, ksize=(33, 33), sigmaX=100, sigmaY=100)
                ii   = i / blur
                ii   = np.where(np.isnan(ii), 0, 1)
                ii   = block_reduce(ii, block_size=kernel_size, func=np.min).astype(np.float32)
                i    = cv2.resize(ii, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST_EXACT)
                i   = (i / np.max(i))
                
                return i

            @staticmethod
            def enhance_text_for_ocr(img, n_times : int = 1):
                for _ in range(n_times):
                    # Step 1: Apply CLAHE to improve local contrast
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
                    img_clahe = clahe.apply(img)
                    
                    # Step 2: Apply Filter to reduce noise (speckle)
                    blurred = cv2.GaussianBlur(img_clahe, (9, 9), 0)
                    
                    # Step 3: Threshold the image to make it binary
                    _, binary_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    # Step 4: Apply 
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                    cleaned_image = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
                    
                    # Step 4: Apply dilation to enhance the text further
                    kernel = np.ones((2, 2), np.uint8)
                    dilated = cv2.dilate(cleaned_image, kernel, iterations=1)
                    
                    # Step 5: Sharpening to make edges clearer
                    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                    sharpened = cv2.filter2D(dilated, -1, sharpen_kernel)
                    img = sharpened

                return sharpened

            @staticmethod
            def get_bounding_box(img, kernel_size : int | float = 256)->np.ndarray:
                if isinstance(kernel_size, float):
                    # get the ratios
                    kernel_size=(int(img.shape[0]*kernel_size), int(img.shape[1]*kernel_size))
                    ii = AppleTextVisualization.isolate_ic(img, kernel_size=kernel_size)
                else:
                    ii = AppleTextVisualization.isolate_ic(img, kernel_size=kernel_size)
                # get bounding box

                (x, y) = ii.shape

                x_upper = 0
                x_lower = x - 1
                y_upper = 0
                y_lower = y - 1

                while x_upper < x:
                    if np.any(ii[x_upper,:] > 0):
                        break
                    else:
                        x_upper += 1
                
                while y_upper < y:
                    if np.any(ii[:,y_upper] > 0):
                        break
                    else:
                        y_upper += 1

                while x_lower > 0:
                    if np.any(ii[x_lower,:] > 0):
                        break
                    else:
                        x_lower -= 1

                while y_lower > 0:
                    if np.any(ii[:,y_lower] > 0):
                        break
                else:
                    y_lower -= 1

                bbox = [x_upper, y_upper, x_lower - x_upper + 1, y_lower - y_upper + 1]
                return bbox
            
        
        if self.active_commands['-apple-ocr']:
            if ocr_mode == 0:
                image_np = np.array(image).mean(axis=-1).astype(np.uint8)
                dataframe, _ = AppleTextVisualization.Image_To_Text(~image_np, rounds=2, minimum_horizontal_size=2048)
            else:
                ocr_instance = AppleOCR(image=image)
                try:
                    dataframe    = ocr_instance.recognize()
                except:
                    dataframe    = None

            if not (dataframe is None):
                return dataframe['Content']
            else:
                return None
            
        img_type = "JPEG"
        iii = BytesIO()
        image.save(iii, format=img_type)
        img_str = base64.b64encode(iii.getvalue()).decode('utf-8')

        if self.active_commands['-llama-vision']:
            from ollama import chat
            from ollama import ChatResponse
            prompt = "You are an OCR agent. Return only the text in the image."

            response: ChatResponse = chat(model='llama3.2-vision', messages=[
            {
                'role': 'user',
                'content': prompt,
                'images' : [img_str]
            },
            ])
            # or access fields directly from the response object
            text = response.message.content
            text = text.split()
            return text

        if self.active_commands['-chat-gpt']:
            from openai import OpenAI

            api_key = os.environ.get("OPENAI_API_KEY")
            client = OpenAI(
                api_key=api_key,  # This is the default and can be omitted
            )
            prompt = "Read all text on the image. Do not use code or OCR, use vision. Return only text. If there is no text, return nothing"

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_str}"},
                            },
                        ],
                    }
                ],
            )

            text = response.choices[0].message.content
            text = text.split()
            return text
        return []

    def add_to_command_tree(self, cmd : str, input : list):
        self.command_tree[cmd] = input

    def print_command_tree(self):
        print(self.command_tree)

    def __bb_core(self, image : Image.Image)->Image.Image:
        image = image.copy()

        # start of model
        bb = BoundingBox(self.active_commands['-model'])

        # compute which cl to use
        cls = []
        ccs = []
        if not (self.active_commands['-overlay-bent-pin'] is None):
            cls.append(0)
            ccs.append(colors[self.active_commands['-overlay-bent-pin']])

        if not (self.active_commands['-overlay-good-pin'] is None):
            cls.append(1)
            ccs.append(colors[self.active_commands['-overlay-good-pin']])

        if not (self.active_commands['-overlay-package'] is None):
            cls.append(2)
            ccs.append(colors[self.active_commands['-overlay-package']])

        if not (self.active_commands['-overlay-text'] is None):
            cls.append(3)
            ccs.append(colors[self.active_commands['-overlay-text']])

        overlay = bb.overlay(image, 10, cls, ccs)

        return overlay
    
    def __text_core(self, image : Image.Image, path : str)->list:
        # no need to run text models
        if self.active_commands['-no-text']:
            return []
        
        pe = PackageExtractor(self.active_commands['-model'])
        cropped, _ = pe.extract_package(image)
        text = []

        # test command using Apple's vision framework for text recognition
        if self.active_commands['-apple-ocr'] or \
           self.active_commands['-chat-gpt']  or \
           self.active_commands['-llama-vision']:
            text = self.runa_ocr(cropped, ocr_mode=0)
            print(text)
            return text

        # apply text recognition here
        text = text_recognition(image, path)

        print(text)
        return text

    def viewer(self, overlay : Image.Image, path : str, text : str | list)->None:
        if not self.active_commands['-show']:
            return
        
        if isinstance(text, str):
            text = [text]

        tt = ""
        for t in text:
            tt += t + ", "

        plt.figure(figsize=[10, 5])
        plt.imshow(overlay)
        plt.title(f"Image from file {path}")
        plt.xlabel(f"Text = {tt}")
        plt.show()

    def save_text(text : str | list, path : str)->None:
        if isinstance(text, str):
            text = [text]

        with open(path, "w") as file:
            for t in text:
                file.write(t + "\n")

    def saver(self, overlay : Image.Image, text : str | list, image_save_path : str, text_save_path : str)->None:
        if not self.active_commands['-save']:
            return
        
        if not self.active_commands['-no-image']:
            overlay.save(image_save_path)

        if not self.active_commands['-no-text']:
            self.save_text(text, text_save_path)

    def parser(self):
        # if overlay commands  are in command tree, set all overlays to None
        for v in ['-overlay-good-pin', '-overlay-bent-pin', '-overlay-package', '-overlay-text']:
            if v in self.command_tree:
                self.active_commands['-overlay-good-pin'] = None
                self.active_commands['-overlay-bent-pin'] = None
                self.active_commands['-overlay-package'] = None
                self.active_commands['-overlay-text'] = None

        for key, value in self.command_tree.items():
            if len(value) == 0:
                self.active_commands[key] = True
            else:
                if value[0] in ["None", "none"]:
                    self.active_commands[key] = None
                else:
                    self.active_commands[key] = value[0]

        if len(self.command_tree) == 0: # default behavior if no commands are given
            self.active_commands['-show'] = True

    def execute(self, path : str):
        t0 = time()
        self.parser()

        image = Image.open(path)
        overlay = self.__bb_core(image)
        text    = self.__text_core(image, path)
        t1 = time()
        print(f'### detector.py: It took {t1 - t0} seconds to execute!')

        self.viewer(overlay, path, text)
        self.saver (overlay, text, self.active_commands['-ifile'], self.active_commands['-tfile'])

if __name__ == '__main__':
    # cmdline parser
    args = sys.argv

    # if it is less than 2 arguments,
    # then return the help menu to the user
    if len(args) < 2:
        help()
        exit(-1)

    filename = args[1]

    de = Detector()
    i = 2
    while i < len(args):
        v = args[i]
        if is_a_command(v):
            cmd = v
            i  += 1 # everytime we get a command or parameter, we increment by 1

            num_inputs = num_parameters(cmd)
            inputs = []
            for j in range(num_inputs):
                inputs.append(args[i])
                i += 1
            
            de.add_to_command_tree(cmd, inputs)
        else:
            print(f"{v} is not a valid command...")
            print("Use -help to see help menu")
            exit(-1)

    de.execute(filename)        # run program
