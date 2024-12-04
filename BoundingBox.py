from PIL import Image
import numpy as np
from ultralytics import YOLO
from matplotlib import pyplot as plt
from DataAugmentator import relative2xy

class BoundingBox:
    def __init__(self, path : str)->None:
        """
            Args:
                path (str): Path to the model

                Example:
                bb = BoundingBox("783-Pin-Detection/runs/detect/train45/weights/best.pt")
        """
        self.path = path
        self.model = YOLO(self.path)

    def __overlay(self, results, image : Image.Image, linewidth : int = 10, cl : int = 0, color : list = [255, 0 ,0]):
        boxes     = results[0].boxes.data.cpu().numpy()
        bent_pins = [det for det in boxes if int(det[-1]) == cl]

        image_np = np.array(image)
        for bent_pin in bent_pins:
            x1, y1, x2, y2 = int(bent_pin[0]), int(bent_pin[1]), int(bent_pin[2]), int(bent_pin[3])
            image_np[y1:y1+linewidth, x1:x2, :] = color
            image_np[y2-linewidth:y2, x1:x2, :] = color
            image_np[y1:y2, x1:x1+linewidth, :] = color
            image_np[y1:y2, x2-linewidth:x2, :] = color

        image = Image.fromarray(image_np)
        return image


    def overlay(self, image : str | Image.Image | np.ndarray, linewidth : int = 10, cl : int | list = 0,
                color : list = [[255, 0, 0]])->Image.Image:
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            image = image.copy()
        else:
            raise ValueError("image must be a file path (str), numpy array, or Image.Image.")
        
        if isinstance(cl, list):
            cls = cl
        elif isinstance(cl, int):
            cls = [cl]
        
        results = self.model.predict(image)
        for i, c in enumerate(cls):
            if isinstance(color, list):
                if isinstance(color[0], list):
                    cc = color[i % len(color)]
                elif isinstance(color[0], int):
                    cc = color
            else:
                raise ValueError("Color must be either a list of three ints [int, int, int],"
                                 "Or a list of list of ints [[int, int, int], ...]")
            
            image = self.__overlay(results, image, linewidth, c, cc)
        return image
        



