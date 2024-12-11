import numpy as np
import cv2
from PIL import Image

def TextEnhancement(img : str | np.ndarray, minimum_horizontal_size : int = 1024, rounds : int = 4, kernel_size : int | float = 64):
    if isinstance(img, str):
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    if isinstance(img, Image.Image):
        img = np.array(img.convert('L'))

    if img.shape[0] < minimum_horizontal_size:
        ratio = minimum_horizontal_size/img.shape[0]
        img   = cv2.resize(img.astype(np.float32), dsize=(int(img.shape[1]*ratio), minimum_horizontal_size)).astype(np.uint8)

    ii = img
    iii = enhance_text_for_ocr(ii, rounds)

    return Image.fromarray(iii)

def enhance_text_for_ocr(img, n_times : int = 1):
    for _ in range(n_times):
        # Step 1: Apply CLAHE to improve local contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        img_clahe = clahe.apply(img)
                    
        # Step 2: Apply Filter to reduce noise (speckle)
        blurred = cv2.GaussianBlur(img_clahe, (9, 9), 0)
                    
        # Step 3: Threshold the image to make it binary
        _, binary_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Step 4: Apply 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned_image = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
                    
        # Step 4: Apply dilation to enhance the text further
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(cleaned_image, kernel, iterations=1)
                    
        # Step 5: Sharpening to make edges clearer
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(dilated, -1, sharpen_kernel)
        img = sharpened

    return sharpened