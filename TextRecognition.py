from BoundingBox import BoundingBox
from PackageExtractor import PackageExtractor
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import subprocess
import ast

#########################################################################
# Implement Text Recognition interface here. detector.py will call this #
# function to recognize text                                            #
#########################################################################
def text_recognition(image : Image.Image, path : str, debug : bool = False)->list | str:

    """
        Args:
            image (Image.Image): A PIL image that is cropped using PackageExtractor.
            path (str): File path to the original image
        Returns:
            list[str] | str: This will be fed into saver(...) and viewer(...)
    """
    out = subprocess.run(
        ['./783-Text-Detection/main.bash',
             path,
             './783-Text-Detection/out',
             './783-Text-Detection/llamadetection/index.ts',
             './783-Text-Detection/main.py',
             './783-Text-Detection/analyze_outputs.py'
         ],
        capture_output=True,
        text=True
        )
    output = str(out)
    outlist = output.split("stdout=")
    string = outlist[1]
    start = string.find('[')
    end = string.find(']', start) + 1
    substring = string[start:end]
    python_list = ast.literal_eval(substring)  # Converts to a Python list
    return python_list

# TESTING
if __name__ == '__main__':
    print(text_recognition(None, '783-Text-Detection/photos/01.png'))
    print(text_recognition(None, '783-Text-Detection/photos/02.png'))
    print(text_recognition(None, '783-Text-Detection/photos/03.png'))
    print(text_recognition(None, '783-Text-Detection/photos/04.png'))
