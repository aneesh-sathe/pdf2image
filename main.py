
from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np
import os


def convert_pdf_to_images(pdf_path, output_folder):
    # Convert PDF to list of images
    images = convert_from_path(pdf_path)
    os.makedirs(output_folder, exist_ok=True)

    for i, image in enumerate(images):
        # Save each image
        image_filename = os.path.join(output_folder, f"page_{i + 1}.png")
        image.save(image_filename, 'PNG')
        print(f"Saved image {image_filename}")

        # Process the image to extract figures
        extract_figures_from_image(image_filename, output_folder, i + 1)


def extract_figures_from_image(image_path, output_folder, page_number):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to get binary image
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on hierarchy
    contours = [contours[i]
                for i in range(len(contours)) if hierarchy[0][i][3] == -1]

    # Combine contours to avoid breaking larger figures
    combined_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        combined_contours.append((x, y, x + w, y + h))

    # Group overlapping contours
    def overlap(a, b):
        return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])

    def combine(a, b):
        return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))

    i = 0
    while i < len(combined_contours):
        j = i + 1
        while j < len(combined_contours):
            if overlap(combined_contours[i], combined_contours[j]):
                combined_contours[i] = combine(
                    combined_contours[i], combined_contours[j])
                combined_contours.pop(j)
            else:
                j += 1
        i += 1

    figure_count = 0
    padding = 10  # HYPERPARAMTER: Change padding to add/remove surrounding area of an image
    for x1, y1, x2, y2 in combined_contours:
        if (x2 - x1) > 50 and (y2 - y1) > 50:
            figure_count += 1

            # Add padding to the bounding box
            x_start = max(x1 - padding, 0)
            y_start = max(y1 - padding, 0)
            x_end = min(x2 + padding, image.shape[1])
            y_end = min(y2 + padding, image.shape[0])

            figure = image[y_start:y_end, x_start:x_end]
            figure_filename = os.path.join(
                output_folder, f"page_{page_number}_figure_{figure_count}.png")
            cv2.imwrite(figure_filename, figure)
            print(f"Extracted figure {figure_filename}")


# Usage example
pdf_path = 'input.pdf'
output_folder = 'output-images'
convert_pdf_to_images(pdf_path, output_folder)
