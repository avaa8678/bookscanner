import cv2
import numpy as np

# Load the image


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

# Set custom window size and display

# Morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)

# Display morphological transformation result

# Edge detection
edges = cv2.Canny(morph, 30, 200, apertureSize=3)

# Display edge detection result

# Hough Line Transform
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=20)

# Visualize detected lines

if lines is None:
    print("No lines detected.")
    exit()

angles = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
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
    img_float = image.astype(np.float32)
    blurred = cv2.GaussianBlur(img_float, kernel_size, sigma)
    high_freq = img_float - blurred
    if threshold > 0:
        low_contrast_mask = np.absolute(high_freq) < threshold
        high_freq[low_contrast_mask] = 0
    sharpened = img_float + amount * high_freq
    return np.clip(sharpened, 0, 255).astype(np.uint8)

sharpened = unsharp_mask(rotated, kernel_size=(9, 9), sigma=2.0, amount=2.0)

# Display the sharpened corrected image
cv2.imshow('Sharpened Corrected Image', sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()
