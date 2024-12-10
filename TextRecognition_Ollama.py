import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you have PyQt installed
from PIL import Image
from PIL import ImageOps
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.colors as mc
import colorsys
import sys
import cv2
import os
import re

def text_recognition(img : Image.Image, path : str):
    from ollama import chat
    from ollama import ChatResponse
    from io import BytesIO
    import base64
    prompt = "You are an OCR agent. Return ONLY text in the image as a comma-seperated list. If there is no text, return the string 'NOTEXT'."

    img = adjust_and_apply_dominant_and_rare_colors(img, amount=0.3, num_colors=2, darken_fraction=0.3, photo_name="")

    img_type = "JPEG"
    iii = BytesIO()
    img.save(iii, format=img_type)
    img_str = base64.b64encode(iii.getvalue()).decode('utf-8')

    response: ChatResponse = chat(model='llama3.2-vision', messages=[
    {
        'role': 'user',
        'content': prompt,
        'images' : [img_str]
    },
    ])
    # or access fields directly from the response object
    text = response.message.content
    if text.find('NOTEXT') != -1:
        return []

    return text

def invert_image(img, output_path=None):
    """
    Inverts the colors of the input image.

    Parameters:
    - image_path: Path to the input image.
    - output_path: Optional path to save the inverted image.

    Displays the inverted image and optionally saves it.
    """
    # Open the image
    
    # Invert the image
    inverted_img = ImageOps.invert(img.convert("RGB"))  # Convert to RGB to ensure compatibility with grayscale images
    
    # Display the inverted image
    # plt.imshow(inverted_img)
    # plt.axis("off")
    # plt.show()

    # Save the inverted image if an output path is provided
    if output_path:
        inverted_img.save(output_path)

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
    img = img.copy().convert('L')  # 'L' mode is grayscale
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

    return adjusted_img

def find_dominant_colors(processed_img, num_colors=5):
    # Flatten the processed image to a 2D array of pixels
    pixels = processed_img.reshape(-1, 1)

    # Use KMeans to find the most dominant grayscale intensities
    kmeans = KMeans(n_clusters=num_colors, random_state=0).fit(pixels)
    dominant_colors = kmeans.cluster_centers_

    return dominant_colors