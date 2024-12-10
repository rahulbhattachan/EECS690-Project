import os
import numpy as np
from PIL import Image
import random
from matplotlib import pyplot as plt

def combineAllBoxes(bboxes : list):
    new_boxes = []
    for boxes in bboxes:
        for box in boxes:
            c, x, y, w, h = \
                box['class'], box['x'], box['y'], box['w'], box['h']
            new_boxes.append({
                'class' : int(c),
                'x' : x, 'y' : y, 'w' : w, 'h' : h
            })
    return new_boxes

def save2file(data, file_path):
    """
    Save a list of dictionaries to a .txt file in the format "class x y w h".

    Parameters:
    - data (list): List of dictionaries with keys 'class', 'x', 'y', 'w', 'h'.
    - file_path (str): The path to the .txt file where the data should be saved.

    Example:
    save_to_txt(data, "output.txt")
    """
    with open(file_path, "w") as f:
        for item in data:
            # Format each line as "class x y w h"
            line = f"{item['class']} {item['x']} {item['y']} {item['w']} {item['h']}\n"
            f.write(line)
    print(f"Data saved to {file_path}")

def relative2xy(sz, pin : dict):
    xcenter = pin['x'] * sz[0]
    ycenter = pin['y'] * sz[1]

    hw      = pin['w'] * sz[0] / 2
    hh      = pin['h'] * sz[1] / 2

    x1 = int(xcenter - hw)
    y1 = int(ycenter - hh)
    x2 = int(xcenter + hw)
    y2 = int(ycenter + hh)

    return x1, y1, x2, y2

def xy2relative(sz, x1, y1, x2, y2):
    a = x1 / sz[0]
    b = y1 / sz[1]
    c = x2 / sz[0]
    d = y2 / sz[1]

    # a, x1 / sz0 = x - w / 2
    # b, y1 / sz1 = y - h / 2
    # c, x2 / sz0 = x + w / 2
    # d, y2 / sz1 = y + h / 2

    # a + c = x - w / 2 + x + w / 2
    # a + c = 2 * x
    # x = (a + c) / 2

    # b + d = y - h / 2 + y + h / 2
    # y = (b + d) / 2

    x = (a + c) / 2
    y = (b + d) / 2
    w = (x - a) * 2
    h = (y - b) * 2
    return x, y, w, h

def get_boxes(boxes : list, cl : int = 0):
    return [
        x for x in boxes if x['class']==cl
    ]

def overlay_good_pins(image_input, good_pins : list):
    """
    Overlays green bounding boxes for good pins (class 1) on the image.

    Args:
        image_path (str): Path to the image.
        label_path (str): Path to the label file.

    Returns:
        None: Displays the image with overlaid bounding boxes.
    """
    if isinstance(image_input, str):
        image = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input)
    elif isinstance(image_input, Image.Image):
        image = image_input.copy()
    else:
        raise ValueError("image must be a file path (str), numpy array, or Image.Image.")
    
    for pin in good_pins:
        x1, y1, x2, y2 = relative2xy(image.size, pin)

        red = np.zeros(shape=(y2 - y1, x2 - x1, 3), dtype=np.uint8)
        red[:,:,0]=255

        r_image = Image.fromarray(red)

        image.paste(r_image, (x1, y1, x2, y2))

    return image

def read_boxes(label_file: str) -> list:
    """
    Reads a .txt label file and extracts bounding boxes with their class.

    Args:
        label_file (str): Path to the label file.

    Returns:
        list: A list of dictionaries containing bounding box data.
              Each dictionary has keys: 'class', 'x', 'y', 'w', 'h'.
    """
    boxes = []
    try:
        with open(label_file, 'r') as file:
            for line in file:
                parts = line.strip().split(' ')
                if len(parts) != 5:
                    print(f"Invalid label format in {label_file}: {line}")
                    continue

                c, x, y, w, h = map(float, parts)
                boxes.append({'class': int(c), 'x': x, 'y': y, 'w': w, 'h': h})

    except FileNotFoundError:
        print(f"Error: The file {label_file} was not found.")
    except Exception as e:
        print(f"Error reading {label_file}: {e}")

    return boxes

class PinRelabeler:
    def __init__(self, images_folder: str, labels_folder: str) -> None:
        """
        Initializes the DataAugmentator and processes the images and labels.
        Crops bent pin regions (class 0) and categorizes them into horizontal and vertical pins.
        """
        self.images_folder = images_folder
        self.labels_folder = labels_folder

    def relabel_pins( self, image, boxes: str | list, package_boxes = None):
        """
        Replaces good pins (class 1) in the image with aggregated bent pins.

        Args:
            image (str | Image.Image | np.ndarray): Input image.
            boxes (list): List of bounding boxes with their properties.
            n (int): Number of bent pins to add.

        Returns:
            tuple: (Modified image, updated good pins, updated bad pins)
                - Image.Image: The modified image with bent pins added.
                - list: The updated list of good pins (remaining after modification).
                - list: The updated list of bad pins (new bent pins added to the image).
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("image must be a file path (str), numpy array, or Image.Image.")
        
        if isinstance(boxes, str):
            boxes = read_boxes(boxes)

        image = image.copy()
        good_pins = get_boxes(boxes, cl=1)
        bad_pins  = get_boxes(boxes, cl=0)

        package_x1, package_y1, package_x2, package_y2 = relative2xy(image.size, get_boxes(boxes, cl=2)[0])


        left_okay_pins = []
        left_bent_pins = []
        right_okay_pins = []
        right_bent_pins = []
        up_okay_pins = []
        up_bent_pins = []
        down_okay_pins = []
        down_bent_pins = []
        for pin in good_pins:
            x1, y1, x2, y2 = relative2xy(image.size, pin)

            # figure orientation of the pin

            # if the package is left or right of the package, it is most likely a horizontal pin
            orientation = None
            if (x1 < package_x1 or x2 > package_x2):
                if (y1 > package_y1 or y2 < package_y2):
                    orientation = 'horizontal'
                else:
                    orientation = 'vertical'
            
            if (y1 < package_y1 or y2 > package_y2):
                if (x1 > package_x1 or x2 < package_y2):
                    orientation = 'vertical'
                else:
                    orientation = 'horizontal'

            if orientation is None:
                orientation = 'vertical' if abs(y2 - y1) > abs(x2 - x1) else 'horizontal'

            if orientation == 'vertical':
                if (y1 < package_y1):       # above package (so it sticks out from above package)
                    mode = 'top'
                    clinfo = 0
                else:                       # below package (so it sticks out below package)
                    mode = 'bottom'
                    clinfo = 1
            else:
                if (x1 < package_x1):       # left of package
                    mode = 'left'
                    clinfo = 2
                else:
                    mode = 'right'
                    clinfo = 3
            
            pin_info = {
                'class' : clinfo,
                'x' : pin['x'],
                'y' : pin['y'],
                'w' : pin['w'],
                'h' : pin['h']
            }

            if mode == 'top':
                up_okay_pins.append(pin_info)
            elif mode == 'bottom':
                down_okay_pins.append(pin_info)
            elif mode == 'left':
                left_okay_pins.append(pin_info)
            elif mode == 'right':
                right_okay_pins.append(pin_info)
            
        for pin in bad_pins:
            x1, y1, x2, y2 = relative2xy(image.size, pin)

            # figure orientation of the pin

            # if the package is left or right of the package, it is most likely a horizontal pin
            orientation = None
            if (x1 < package_x1 or x2 > package_x2):
                if (y1 > package_y1 or y2 < package_y2):
                    orientation = 'horizontal'
                else:
                    orientation = 'vertical'
            
            if (y1 < package_y1 or y2 > package_y2):
                if (x1 > package_x1 or x2 < package_y2):
                    orientation = 'vertical'
                else:
                    orientation = 'horizontal'

            if orientation is None:
                orientation = 'vertical' if abs(y2 - y1) > abs(x2 - x1) else 'horizontal'

            if orientation == 'vertical':
                if (y1 < package_y1):       # above package (so it sticks out from above package)
                    mode = 'top'
                    clinfo = 4
                else:                       # below package (so it sticks out below package)
                    mode = 'bottom'
                    clinfo = 5
            else:
                if (x1 < package_x1):       # left of package
                    mode = 'left'
                    clinfo = 6
                else:
                    mode = 'right'
                    clinfo = 7
            
            pin_info = {
                'class' : clinfo,
                'x' : pin['x'],
                'y' : pin['y'],
                'w' : pin['w'],
                'h' : pin['h']
            }

            if mode == 'top':
                up_bent_pins.append(pin_info)
            elif mode == 'bottom':
                down_bent_pins.append(pin_info)
            elif mode == 'left':
                left_bent_pins.append(pin_info)
            elif mode == 'right':
                right_bent_pins.append(pin_info)
            
        return left_okay_pins, left_bent_pins,      \
               right_okay_pins, right_bent_pins,    \
               up_okay_pins, up_bent_pins,          \
               down_okay_pins, down_bent_pins