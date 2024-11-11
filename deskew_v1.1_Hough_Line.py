import cv2
import numpy as np

# Load the image
image = cv2.imread(r'C:\Users\james\Downloads\tiled_2.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Equalize the histogram to improve contrast
gray = cv2.equalizeHist(gray)

# Apply Gaussian Blur
gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply median blur to reduce noise
gray_blurred = cv2.medianBlur(gray_blurred, 3)

# Apply Otsu's thresholding
_, thresh = cv2.threshold(gray_blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imshow('Thresholded Image', thresh)
cv2.waitKey(0)

# Morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)

# After morphological operations
cv2.imshow('Morphological Transformation', morph)
cv2.waitKey(0)

# Edge detection
edges = cv2.Canny(morph, 30, 200, apertureSize=3)

# After edge detection
cv2.imshow('Edge Detection', edges)
cv2.waitKey(0)

# Hough Line Transform
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=20)

# Visualize detected lines
line_img = image.copy()
for line in lines:
    x1, y1, x2, y2 = line[0]
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if length > 100:  # Minimum line length threshold
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imshow('Detected Lines', line_img)
cv2.waitKey(0)

if lines is None:
    print("No lines detected.")
    exit()

angles = []

for line in lines:
    x1, y1, x2, y2 = line[0]
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    # Filter angles
    if -45 < angle < 45:
        angles.append(angle)

if not angles:
    print("No valid angles found.")
    exit()

# Compute the median angle
median_angle = np.median(angles)

# Rotate the image to correct the skew
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

# Sharpen the rotated image using Unsharp Masking
def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0):
    # Convert to float32 to prevent data loss
    img_float = image.astype(np.float32)
    
    # Blur the image
    blurred = cv2.GaussianBlur(img_float, kernel_size, sigma)
    
    # Calculate the high-frequency components
    high_freq = img_float - blurred
    
    # Apply threshold if needed
    if threshold > 0:
        low_contrast_mask = np.absolute(high_freq) < threshold
        high_freq[low_contrast_mask] = 0
    
    # Amplify the high frequencies
    sharpened = img_float + amount * high_freq
    
    # Clip values to valid range and convert back to uint8
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened

sharpened = unsharp_mask(rotated, kernel_size=(9, 9), sigma=2.0, amount=2.0)

# Display the sharpened corrected image
cv2.imshow('Sharpened Corrected Image', sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()
