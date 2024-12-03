import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, image_path, label_path, train : bool = False, transform=None):
        """
        Args:
            folder_path (str): Path to the dataset folder.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_path = image_path
        self.label_path = label_path
        self.transform = transform
        self.image_paths = []
        self.label_paths = []
        self.images = []
        self.pins   = []

        # Load images
        for filename in sorted(os.listdir(self.image_path)):
            print(filename)
            print(type(filename))
            if filename.endswith(".png") or filename.endswidth(".jpg"):
                self.image_paths.append(os.path.join(image_path, filename))

        # Load labels
        for filename in sorted(os.listdir(self.label_path)):
            if filename.endswith(".txt"):
                self.label_paths.append(os.path.join(label_path, filename))

        # For each image and its corresponding labels
        for image_path, label_path in zip(self.image_paths, self.label_paths):
            image  = np.array(Image.open(image_path)).squeeze().sum(axis=-1)/(255 * 3)
            labels = self.open_labels(label_path)

            h, w   = image.shape

            # We now compute for each label
            for i in range(len(labels)):
                label = labels[i]
                p, x, y, width, height = int(label[0]), int(label[1] * w), int(label[2] * h), int(label[3] * w), int(label[4] * h)

                self.images.append(image[
                    y:y+height, x:x+width
                ])

                self.pins.append(
                    p
                )
        
        self.images = torch.tensor(self.images, dtype=torch.float32)
        self.pins   = torch.tensor(self.pins, dtype=torch.float32)

    def open_labels(self, path):
        labels = []
        with open(path, 'r') as file:
            for line in file:
                u = line.split()
                if int(u[0]) == 1 or int(u[0]) == 0:
                    labels.append([
                        float(x) for x in u
                    ])
        return labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image_array = np.array(image)
        image_tensor = torch.tensor(image_array, dtype=torch.float32)

        # Add a channel dimension if image is grayscale
        if len(image_tensor.shape) == 2:
            image_tensor = image_tensor.unsqueeze(0)

        # Normalize image
        image_tensor = image_tensor / 255.0  # Scale to [0, 1]

        # Apply any specified transformations
        if self.transform:
            image_tensor = self.transform(image_tensor)

        # Get the corresponding temperature
        temperature = self.temperatures[idx]

        return image_tensor, temperature