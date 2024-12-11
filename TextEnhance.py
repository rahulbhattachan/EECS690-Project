import numpy as np
import cv2
from PIL import Image, ImageEnhance


def TextEnhancement(image, min_height):
    image = ImageEnhance.Contrast(image).enhance(6)
    def ensure_minimum_height(image, min_height):
        """
        Ensure the image's height is at least `min_height`. Resizes if necessary.
        
        Parameters:
            image (PIL.Image): The input image.
            min_height (int): The minimum height required.
        
        Returns:
            PIL.Image: The resized (or original) image.
        """
        # Get current dimensions
        width, height = image.size
        
        # Check if height is below the minimum
        if height < min_height:
            # Calculate the new width to maintain aspect ratio
            new_width = int((min_height / height) * width)
            # Resize the image
            image = image.resize((new_width, min_height), Image.Resampling.LANCZOS)
        
        return image

    image = ensure_minimum_height(image, min_height)
    image = np.array(image)

    blur = cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=33, sigmaY=33)
    image = cv2.divide(image, blur, scale=255)

    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Apply adaptive thresholding to enhance text visibility
    #image = cv2.adaptiveThreshold(
    #    image,
    #    255,
    #    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #    cv2.THRESH_BINARY,
    #    251,
    #    5,
    #)

    # Define a kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, None, None, None, 8)
    areas = stats[1:,cv2.CC_STAT_AREA]

    results = np.zeros_like(image)
    for i in range(0, nlabels-1):
        if areas[i] >= 600:
            results[labels == i + 1] = 255

    image = Image.fromarray(results)
    return image


def TextEnhancement_Old(image, min_height):
    """
    Process the input image to enhance text visibility. Includes resizing, sharpening,
    contrast adjustment, adaptive thresholding, denoising, and morphological operations.

    Parameters:
        image (PIL.Image): The input image.
        min_height (int): The minimum height for resizing.

    Returns:
        PIL.Image: The processed image.
    """
    # Step 1: Adjust contrast
    image = ImageEnhance.Contrast(image).enhance(3)

    # Step 2: Ensure minimum height
    def ensure_minimum_height(image, min_height):
        width, height = image.size
        if height < min_height:
            new_width = int((min_height / height) * width)
            image = image.resize((new_width, min_height), Image.Resampling.LANCZOS)
        return image

    image = ensure_minimum_height(image, min_height)
    image = np.array(image)

    # Step 3: Sharpen the image
    sharpening_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    image = cv2.filter2D(image, -1, sharpening_kernel)

    # Step 4: Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Step 5: Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
    image = clahe.apply(blurred)

    # Step 6: Apply adaptive thresholding
    adaptive_threshold = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        101,
        1
    )

    # Step 7: Morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    image = cv2.morphologyEx(adaptive_threshold, cv2.MORPH_OPEN, kernel)

    # Step 8: Connected component analysis to filter small regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)
    min_area = 50  # Minimum area threshold
    cleaned_image = np.zeros_like(image)

    for i in range(1, num_labels):  # Skip background label 0
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned_image[labels == i] = 255

    # Step 9: Final morphological cleaning
    image = cv2.morphologyEx(cleaned_image, cv2.MORPH_OPEN, kernel)

    # Convert the result back to a Pillow image
    return Image.fromarray(image)
