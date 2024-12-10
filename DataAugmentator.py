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

class DataAugmentator:
    def __init__(self, images_folder: str, labels_folder: str) -> None:
        """
        Initializes the DataAugmentator and processes the images and labels.
        Crops bent pin regions (class 0) and categorizes them into horizontal and vertical pins.
        """
        self.images_folder = images_folder
        self.labels_folder = labels_folder

    def add_bent_pins(self, image, boxes: str | list, n: int,
                      extend : int | tuple | float = 30,
                      angle  : int | tuple = 20,
                      cls_in : int = 1,
                      cls_out : int = 0,
                      package_boxes = None):
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
        class_in_pins = get_boxes(boxes, cl=cls_in)

        pins_to_replace   = random.sample(class_in_pins, min(n, len(class_in_pins)))
        updated_class_in_pins = [pin for pin in class_in_pins if pin not in pins_to_replace]
        updated_class_out_pins  = []

        if isinstance(extend, int):
            mina = 0
            maxa = extend
        elif isinstance(extend, tuple):
            mina = extend[0]
            maxa = extend[1]
        elif isinstance(extend, float):
            pass
        else:
            raise ValueError("Extend must be int, tuple, or float")

        if isinstance(angle, int):
            minb = -angle
            maxb =  angle
        else:
            minb = angle[0]
            maxb = angle[1]

        package_x1, package_y1, package_x2, package_y2 = relative2xy(image.size, get_boxes(boxes if package_boxes == None else package_boxes, cl=2)[0])
        package = image.crop((package_x1, package_y1, package_x2, package_y2))
        for pin in pins_to_replace:
            x1, y1, x2, y2 = relative2xy(image.size, pin)
            orig_pin = image.crop((x1, y1, x2, y2))

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

            if isinstance(extend, float):
                mina = 0
                maxa = int((extend - 1.0) * (abs(y2 - y1) if orientation == 'vertical' else abs(x2 - x1)))

            angle      = 0
            extend_pin = 0
    
            while angle == 0 or extend_pin == 0:
                extend_pin  = random.randint(mina, maxa)   # +/- pixels
                angle       = random.randint(minb, maxb)     # +/- deg

            # if it is a vertical pin, we need to check if the pin to modify is top or bottom
            orig_pin_np = np.array(orig_pin)
            mode = None
            if orientation == 'vertical':
                if (y1 < package_y1):       # above package (so it sticks out from above package)
                    mode = 'top'
                    new_x1, new_y1, new_x2, new_y2 = x1, y1 - extend_pin, x2, y2
                    sampled_color = orig_pin_np[:5,:,:].mean(axis=(0,1))
                else:                       # below package (so it sticks out below package)
                    new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2 + extend_pin
                    mode = 'bottom'
                    sampled_color = orig_pin_np[-5:-1,:,:].mean(axis=(0,1))
            else:
                if (x1 < package_x1):       # left of package
                    mode = 'left'
                    new_x1, new_y1, new_x2, new_y2 = x1 - extend_pin, y1, x2, y2
                    sampled_color = orig_pin_np[:,:5,:].mean(axis=(0,1))
                else:
                    mode = 'right'
                    new_x1, new_y1, new_x2, new_y2 = x1, y1, x2 + extend_pin, y2
                    sampled_color = orig_pin_np[:,-5:-1,:].mean(axis=(0,1))

            bent_pin = orig_pin.resize((new_x2 - new_x1, new_y2 - new_y1))
            bent_pin = bent_pin.rotate(angle=angle, fillcolor=(int(sampled_color[0]), int(sampled_color[1]), int(sampled_color[2])))

            reduce = int(0.01 * angle)
            if reduce <= 0:
                reduce = 1

            bent_pin_np = np.array(bent_pin)

            if mode == 'top':
                shrink_x1, shrink_y1, shrink_x2, shrink_y2 = x1, y1 + reduce , x2, y2
                bent_pin_np=bent_pin_np[0:-reduce,:,:]
            elif mode == 'bottom':
                shrink_x1, shrink_y1, shrink_x2, shrink_y2 = x1, y1, x2, y2 - reduce
                bent_pin_np=bent_pin_np[reduce:-1,:,:]
            elif mode == 'left':
                shrink_x1, shrink_y1, shrink_x2, shrink_y2 = x1 + reduce, y1, x2, y2
                bent_pin_np=bent_pin_np[:,0:-reduce,:]
            elif mode == 'right':
                shrink_x1, shrink_y1, shrink_x2, shrink_y2 = x1, y1, x2 - reduce, y2
                bent_pin_np=bent_pin_np[:,reduce:-1,:]
            
            bent_pin = Image.fromarray(bent_pin_np)

            # Replace the good pin with the composite bent pin
            image.paste(bent_pin, (shrink_x1, shrink_y1))

            x, y, w, h =xy2relative(image.size, new_x1, new_y1, new_x2, new_y2)
            updated_class_out_pins.append({
                'class': cls_out,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
            })
        
        image.paste(package, (package_x1, package_y1))

        return image, updated_class_in_pins, updated_class_out_pins