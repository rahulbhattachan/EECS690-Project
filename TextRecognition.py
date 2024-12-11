from BoundingBox import BoundingBox
import os
from PackageExtractor import PackageExtractor
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import subprocess
import ast
from dotenv import load_dotenv

#########################################################################
# Implement Text Recognition interface here. detector.py will call this #
# function to recognize text                                            #
#########################################################################
def text_recognition(image : Image.Image, path : str, debug : bool = False, breakpoint : bool = False)->list | str:

    """
        Args:
            image (Image.Image): A PIL image that is cropped using PackageExtractor.
            path (str): File path to the original image
        Returns:
            list[str] | str: This will be fed into saver(...) and viewer(...)
    """
    current_dir = os.getcwd()
    path = os.path.join(current_dir, path)

    env_path = os.path.join(current_dir, '783-Text-Detection/.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
    paths_to_check = [
        './783-Text-Detection/main.bash',
        path,  # Ensure the passed-in 'path' exists
        './783-Text-Detection/out',
        './783-Text-Detection/llamadetection/index.ts',
        './783-Text-Detection/main.py',
        './783-Text-Detection/analyze_outputs.py'
    ]
    for p in paths_to_check:
        full_path = os.path.join(os.getcwd(), p)
        if not os.path.exists(full_path):
            print(f"File not found: {full_path}")

    try:
        result =  subprocess.run(
        ['./783-Text-Detection/main.bash',
         path,
         './783-Text-Detection/out',
         './783-Text-Detection/llamadetection/index.ts',
         './783-Text-Detection/main.py',
         './783-Text-Detection/analyze_outputs.py'
         ],
        capture_output=True,
        text=True,
        check=True,
        env=os.environ  # Ensure the current environment is passed
        )
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")

    output = str(result)
    outlist = output.split("stdout=")
    string = outlist[1]
    start = string.find('[')
    end = string.find(']', start) + 1
    substring = string[start:end]
    python_list = ast.literal_eval(substring)  # Converts to a Python list
    print(python_list)
    return python_list
