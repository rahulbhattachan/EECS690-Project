# Text and Bent Pin Detection

This repository contains two models: 
- 783-Pin-Detection
- 783-Text-Detection

783-Pin-Detection implements a transfer-trained YOLOv11 model. The dataset trained uses 18 non-augmentated images and 18 augmentated images (see DataAugmentator.ipynb for more details). There are 9 validation images. These images have undergone image augmentation.

783-Text-Detection implements a Llama based OCR agent that extracts text from an image and returns as plaintext.

You can call detector.py to process an image, or look into Detector.ipynb to see already-ran results.

Example usage:

    python3 detector.py <path-to-your-image> -show
    python3 detector.py <path-to-your-image> -show -no-text
    python3 detector.py <path-to-your-image> -save
    python3 detector.py <path-to-your-image> -save -ifile <path-to-output-image-file> -tfile <path-to-output-text-file>
