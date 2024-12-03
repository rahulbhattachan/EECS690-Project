import os
import numpy as np
from PIL import Image
import random

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
        self.horizontal_bent_pins = []  # Stores cropped regions of horizontal bent pins as Image.Image
        self.vertical_bent_pins = []    # Stores cropped regions of vertical bent pins as Image.Image
        self.process_bent_pins()

    def process_bent_pins(self) -> None:
        """
        Processes all images and labels in the specified folders, extracts bent pins, 
        and categorizes them into horizontal and vertical pins.
        """
        image_files = [f for f in os.listdir(self.images_folder) if f.endswith(('.png', '.jpg'))]

        for image_file in image_files:
            image_path = os.path.join(self.images_folder, image_file)
            image = Image.open(image_path).convert('RGB')
            image_width, image_height = image.size

            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(self.labels_folder, label_file)

            boxes = read_boxes(label_path)

            for box in boxes:
                if box['class'] == 0:  # Bent pins
                    x1, y1, x2, y2 = relative2xy(image.size, box)

                    cropped_region = image.crop((x1, y1, x2, y2))

                    if (y2 - y1) > (x2 - x1):
                        self.vertical_bent_pins.append(cropped_region)
                    else:
                        self.horizontal_bent_pins.append(cropped_region)

    def add_bent_pins(self, image, boxes: list, n: int, alpha : float = 0.5):
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

        image = image.copy()
        good_pins = [x for x in boxes if x['class'] == 1]

        for pin in good_pins:
            pin['orientation'] = 'vertical' if pin['h'] > pin['w'] else 'horizontal'

        pins_to_replace = random.sample(good_pins, min(n, len(good_pins)))
        updated_good_pins = [pin for pin in good_pins if pin not in pins_to_replace]
        updated_bad_pins = []

        for pin in pins_to_replace:
            x1, y1, x2, y2 = relative2xy(image.size, pin)

            if pin['orientation'] == 'vertical':
                select   = random.randint(0, len(self.vertical_bent_pins)-1)
                bent_pin = self.vertical_bent_pins[select]
            else:
                select   = random.randint(0, len(self.horizontal_bent_pins)-1)
                bent_pin = self.horizontal_bent_pins[select]

            bent_pin = bent_pin.resize((x2 - x1, y2 - y1))
            orig_pin = image.crop((x1, y1, x2, y2))

            # blend with original pins
            # Blend the original pin region with the bent pin
            blended_pin = Image.blend(orig_pin, bent_pin, alpha)


            # Replace the good pin with the composite bent pin
            image.paste(blended_pin, (x1, y1))

            updated_bad_pins.append({
                'class': 0,
                'x': pin['x'],
                'y': pin['y'],
                'w': pin['w'],
                'h': pin['h']
            })

        return image, updated_good_pins, updated_bad_pins