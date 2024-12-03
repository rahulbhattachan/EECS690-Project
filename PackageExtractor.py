from ultralytics import YOLO
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

class PackageExtractor:
    def __init__(self, path : str)->None:
        """
            Args:
                path (str): Path to the model

                Example:
                pe = PackageExtractor("783-Pin-Detection/runs/detect/train45/weights/best.pt")
        """
        self.path = path
        self.model = YOLO(self.path)

    def extract_package(self, image_input : str | Image.Image | np.ndarray)->Image.Image:
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError("image_input must be either a valid path (str), Image (Image.Image), or Numpy array (np.ndarray)")
        
        width, height = image.size
        
        results = self.model.predict(image, verbose=False)

        detections = results[0].boxes.data.cpu().numpy()

        package_detections = [det for det in detections if int(det[-1]) == 2]
        
        if not package_detections or len(package_detections) == 0:
            return None
        
        largest_package = min(package_detections, key=lambda x : (x[2]*x[3]))
        x1, y1, x2, y2 = map(float, largest_package[0:4])

        x1 = x1 if x1 < width else width
        y1 = y1 if y1 < height else height
        x2 = x2 if x2 < width else width
        y2 = y2 if y2 < height else height

        cropped = image.crop((x1, y1, x2, y2))
        return cropped, results[0]


if __name__ == '__main__':
    pe = PackageExtractor('./783-Pin-Detection/runs/detect/train45/weights/best.pt')

    filenames = ['./783-Pin-Detection/datasets/ic-dataset/images/val/00.png',
                 './783-Pin-Detection/datasets/ic-dataset/images/val/03.png',
                 './783-Pin-Detection/datasets/ic-dataset/images/val/09.png',
                 './783-Pin-Detection/datasets/ic-dataset/images/val/29.jpg',
                 './783-Pin-Detection/datasets/ic-dataset/images/train/02.png',
                 './783-Pin-Detection/datasets/ic-dataset/images/train/04.png',
    ]

    n = len(filenames)
    for i, filename in enumerate(filenames):
        image    = Image.open(filename).convert('RGB')
        cropped, result  = pe.extract_package(image)
        assert cropped != None

        plt.figure(figsize=[10, 5])
        plt.subplot(1,3, 1)
        plt.imshow(image)
        plt.title("Original")

        plt.subplot(1,3, 2)
        plt.imshow(result.plot())
        plt.title("Bounding boxes")

        plt.subplot(1,3, 3)
        plt.imshow(cropped)
        plt.title("Cropped")

        plt.tight_layout()
        plt.show()