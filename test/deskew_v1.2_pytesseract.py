import cv2
import numpy as np
from PIL import Image
import pytesseract
import re
import tempfile
import os

# Specify the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the image
image = cv2.imread(r'C:\Users\james\OneDrive\Documents\Coding\Bookscanner\bookscanner\examples\tiled.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to create a binary image
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 2)

# Optionally, apply a slight blur to reduce noise
blurred = cv2.GaussianBlur(thresh, (5, 5), 0)

# Convert to PIL Image
pil_image = Image.fromarray(blurred)

# Save the image to a temporary file with DPI set
temp_filename = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
pil_image.save(temp_filename, dpi=(300, 300))

try:
    # Use pytesseract to detect orientation and script detection (OSD)
    osd = pytesseract.image_to_osd(temp_filename)
    print("OSD Output:")
    print(osd)

    # Extract the rotation angle using regular expressions
    angle_match = re.search(r'Rotate: (\-?\d+)', osd)
    if angle_match:
        angle = int(angle_match.group(1))
    else:
        angle = 0
    print(f"Detected angle: {angle} degrees")

    # Rotate the image to correct the skew
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    if angle != 0:
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    else:
        rotated = image.copy()

    # Display the rotated image
    cv2.imshow('Corrected Image', rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except pytesseract.TesseractError as e:
    print(f"TesseractError: {e}")

finally:
    # Clean up the temporary file
    os.remove(temp_filename)
