import numpy as np
import cv2
from PIL import Image, ImageEnhance
from sklearn.cluster import KMeans
import matplotlib.colors as mc
from matplotlib import pyplot as plt
import colorsys

def text_recognition_forced(image : Image.Image, path : str, debug : bool = False, breakpoint : bool = False, no_adjustment : bool = False)->list | str:
    from apple_ocr.ocr import OCR as AppleOCR

    new_height = 2048
    new_width  = int(image.width/image.height * new_height)
    image = image.resize((new_width, new_height))

    if debug:
        plt.imshow(image)
        plt.show()

    if no_adjustment == False:
        image = adjust_and_apply_dominant_and_rare_colors(image, amount=0.3, num_colors=2, darken_fraction=0.3)

    if debug:
        plt.imshow(image)
        plt.show()

    ocr_instance = AppleOCR(image=image)
    try:
        dataframe    = ocr_instance.recognize()
    except:
        dataframe    = None

    if not (dataframe is None):
        return dataframe['Content']
    else:
        return []

def TextEnhancement(image, min_height):
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')
    plt.show()
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

    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    blur = cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=33, sigmaY=33)
    image = cv2.divide(image, blur, scale=255)

    # Define a kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, None, None, None, 8)
    areas = stats[1:,cv2.CC_STAT_AREA]

    results = np.zeros_like(image)
    for i in range(0, nlabels-1):
        if areas[i] >= 50:
            results[labels == i + 1] = 255

    image = cv2.dilate(results, kernel=(3, 3))

    #image = cv2.GaussianBlur(image, ksize=(21, 21), sigmaX=33, sigmaY=33)
    image = Image.fromarray(image)
    return image

def TextEnhancement_old(image, min_height):
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

    # Define a kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, None, None, None, 8)
    areas = stats[1:,cv2.CC_STAT_AREA]

    results = np.zeros_like(image)
    for i in range(0, nlabels-1):
        if areas[i] >= 300:
            results[labels == i + 1] = 255

    #image = cv2.GaussianBlur(image, ksize=(21, 21), sigmaX=33, sigmaY=33)
    image = Image.fromarray(results)
    return image


def adjust_color(color, amount=0.5, lighten=True):
    """Adjusts the brightness of the color based on whether to lighten or darken."""
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    if lighten:
        return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
    else:
        return colorsys.hls_to_rgb(c[0], c[1] * (1 - amount), c[2])

def adjust_and_apply_dominant_and_rare_colors(img : Image.Image, amount=0.5, num_colors=5, darken_fraction=0.3, inverted=False, photo_name=""):
    """
    Loads an image, converts it to grayscale, finds dominant and rare colors,
    lightens the dominant colors, and darkens the rare colors.
    
    Parameters:
    - image_path: Path to the input image.
    - amount: Brightness adjustment factor.
    - num_colors: Number of dominant colors to extract from the image.
    - darken_fraction: Fraction of colors to be darkened (less frequent colors).
    
    Displays:
    - The resulting image with adjusted dominant and rare colors.
    """
    # Load the image and convert it to grayscale
    img_np = np.array(img)
    
    # Reshape the image to a 2D array of pixels
    pixels = img_np.reshape(-1, 1)  # Each pixel has a single intensity value in grayscale

    # Use KMeans to find the most dominant grayscale intensities
    kmeans = KMeans(n_clusters=num_colors, random_state=0).fit(pixels)
    dominant_colors = kmeans.cluster_centers_
    color_counts = np.bincount(kmeans.labels_)

    # Sort colors by frequency
    sorted_indices = np.argsort(color_counts)
    most_common_indices = sorted_indices[-int(num_colors * (1 - darken_fraction)):]  # Top fraction to lighten
    least_common_indices = sorted_indices[:int(num_colors * darken_fraction)]  # Bottom fraction to darken

    # Adjust colors: lighten common colors, darken rare colors
    adjusted_colors = []
    for i, color in enumerate(dominant_colors):
        color_rgb = (color[0] / 255.0, color[0] / 255.0, color[0] / 255.0)  # Convert grayscale to RGB-like format
        if i in most_common_indices:
            adjusted_color = adjust_color(color_rgb, amount, lighten=True)
        else:
            adjusted_color = adjust_color(color_rgb, amount, lighten=False)
        adjusted_colors.append(tuple(int(c * 255) for c in adjusted_color))  # Convert back to [0, 255]

    # Create a mask for each dominant color and apply the adjusted color
    pixels_colored = np.repeat(pixels, 3, axis=1)  # Duplicate grayscale values to RGB format
    for i, original_color in enumerate(dominant_colors):
        new_color = adjusted_colors[i]
        mask = np.isclose(pixels.flatten(), original_color[0], atol=10)
        pixels_colored[mask] = new_color

    # Reshape pixels back to the original image shape and convert to RGB
    adjusted_img_np = pixels_colored.reshape(img_np.shape[0], img_np.shape[1], 3)
    adjusted_img = Image.fromarray(adjusted_img_np.astype('uint8'), 'RGB')

    if inverted:
        adjusted_img = ~adjusted_img
    
    return adjusted_img

def find_dominant_colors(processed_img, num_colors=5):
    # Flatten the processed image to a 2D array of pixels
    pixels = processed_img.reshape(-1, 1)

    # Use KMeans to find the most dominant grayscale intensities
    kmeans = KMeans(n_clusters=num_colors, random_state=0).fit(pixels)
    dominant_colors = kmeans.cluster_centers_

    return dominant_colors
