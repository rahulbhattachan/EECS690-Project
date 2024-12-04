from BoundingBox import BoundingBox
from PackageExtractor import PackageExtractor
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

#########################################################################
# Implement Text Recognition interface here. detector.py will call this #
# function to recognize text                                            #
#########################################################################
def text_recognition(image : Image.Image, path : str)->list | str:
    """
        Args:
            image (Image.Image): A PIL image that is cropped using PackageExtractor.
            path (str): File path to the original image
        Returns:
            list[str] | str: This will be fed into saver(...) and viewer(...)
    """
    return []