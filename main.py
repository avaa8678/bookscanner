#!/usr/bin/env python3
import os
from datetime import datetime
from PIL import Image
import cv2
import hough_line         # Your Hough_Line module
import warp_perspective  # Import your module

# Base folder for results
base_results = "C:/Users/james/OneDrive/Documents/Coding/Bookscanner/bookscanner/output"

# Folder containing the raw scanned images
input_folder = "C:/Users/james/OneDrive/Documents/Coding/Bookscanner/bookscanner/raw"  # Update this path to where your original images are stored

# Create a new output folder using the current timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = os.path.join(base_results, timestamp)
os.makedirs(output_folder, exist_ok=True)
print(f"Output folder created: {output_folder}")

# List all image files (jpg, jpeg, png) in the input folder
image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
if not image_files:
    print("No images found in the input folder.")
    exit(1)

# Ask user which processing method to use
choice = input("Press 1 for Hough Line processing or 2 for Warp Perspective processing: ").strip()

# List to hold paths to the split images for PDF creation
split_image_paths = []
page_counter = 1

for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading {image_file}")
        continue

    # Process the image using the selected method
    if choice == "1":
        result_img = hough_line.process_image(image)
    elif choice == "2":
        result_img = warp_perspective.process_image(image)
    else:
        print("Invalid selection. Exiting.")
        exit(1)

    if result_img is None:
        print(f"Transform failed for {image_file}. Skipping...")
        continue

    # Split the warped image into left and right halves.
    height, width = result_img.shape[:2]
    half_width = width // 2
    left_img = result_img[:, :half_width]
    right_img = result_img[:, half_width:]

    # Define filenames for the left and right halves
    left_filename = f"img_{page_counter}_1.jpg"
    right_filename = f"img_{page_counter}_2.jpg"

    left_output_path = os.path.join(output_folder, left_filename)
    right_output_path = os.path.join(output_folder, right_filename)

    # Save the split images
    cv2.imwrite(left_output_path, left_img)
    cv2.imwrite(right_output_path, right_img)

    print(f"Saved {left_output_path} and {right_output_path}")

    # Add to list for PDF conversion
    split_image_paths.extend([left_output_path, right_output_path])
    page_counter += 1

# Create a PDF file from the split images
pdf_filename = f"scanned_{timestamp}.pdf"
pdf_output_path = os.path.join(base_results, pdf_filename)

# Convert images to RGB (if needed) and add them to the PDF list
pdf_images = []
for path in sorted(split_image_paths):
    try:
        image = Image.open(path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        pdf_images.append(image)
    except Exception as e:
        print(f"Error processing {path} for PDF: {e}")

if pdf_images:
    # Save PDF with all pages (each image becomes one page)
    pdf_images[0].save(pdf_output_path, save_all=True, append_images=pdf_images[1:])
    print(f"PDF created: {pdf_output_path}")
else:
    print("No images available for PDF creation.")