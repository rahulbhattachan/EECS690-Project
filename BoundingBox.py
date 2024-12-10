from PIL import Image
import numpy as np
from ultralytics import YOLO
from matplotlib import pyplot as plt
from DataAugmentator import relative2xy, get_boxes, xy2relative

def is_overlap_greater_than_threshold(rect1, rect2, threshold):
    """
    Check if the overlapping area of two rectangles is greater than a threshold.

    Parameters:
        rect1 (tuple): (x1, y1, x2, y2) for rectangle 1
        rect2 (tuple): (x1, y1, x2, y2) for rectangle 2
        threshold (float): Minimum overlap area required to return True

    Returns:
        bool: True if overlap area > threshold, False otherwise
    """
    # Unpack rectangle coordinates
    x1_1, y1_1, x2_1, y2_1 = rect1
    x1_2, y1_2, x2_2, y2_2 = rect2

    # Calculate overlap rectangle
    overlap_x1 = max(x1_1, x1_2)
    overlap_y1 = max(y1_1, y1_2)
    overlap_x2 = min(x2_1, x2_2)
    overlap_y2 = min(y2_1, y2_2)

    # Check if there is an overlap
    if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
        # Calculate overlap area
        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
        return overlap_area > threshold
    else:
        return False  # No overlap

def remove_overlaps(sz, ep : list, rem : list, ratio : float = 0.75)->list:
    updated_rem = []
    for r in rem:
        flag = False
        for p in ep:
            if is_overlap_greater_than_threshold(relative2xy(r), relative2xy(p), ratio):
                flag = True
        if not flag:
            updated_rem.append(r)
    return updated_rem


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

    def __overlay(self, results, image : Image.Image, linewidth : int = 10, cl : int = 0, color : list = [255, 0 ,0], ov_det = []):
        boxes     = results[0].boxes.data.cpu().numpy()
        bent_pins = [det for det in boxes if int(det[-1]) == cl]

        if len(ov_det) > 0 and cl == 1 and ov_det[0] == 0:
            ep        = [det for det in boxes if int(det[-1]) == ov_det[0]]
            updated_bent_pins = []
            for r in bent_pins:
                flag = False
                for p in ep:
                    if is_overlap_greater_than_threshold((int(r[0]), int(r[1]), int(r[2]), int(r[3])),
                                                         (int(p[0]), int(p[1]), int(p[2]), int(p[3])), 0.4):
                        flag = True
                if not flag:
                    updated_bent_pins.append(r)
            bent_pins = updated_bent_pins

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
                color : list = [[255, 0, 0]],
                ov_det = [])->Image.Image:
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
            
            image = self.__overlay(results, image, linewidth, c, cc, ov_det)
        return image
        



