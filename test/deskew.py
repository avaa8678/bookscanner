import cv2
import numpy as np

# Load the image
image = cv2.imread(r'C:\Users\james\OneDrive\Documents\Coding\Bookscanner\bookscanner\examples\tiled_2.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Equalize the histogram to improve contrast
gray = cv2.equalizeHist(gray)

# Apply Gaussian Blur
gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply adaptive thresholding
thresh = cv2.adaptiveThreshold(gray_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 10)

# Morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Edge detection
edges = cv2.Canny(morph, 50, 150, apertureSize=3)

# Hough Line Transform
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=20)

# # Visualize detected lines
# line_img = image.copy()
# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
# cv2.imshow('Detected Lines', line_img)
# cv2.waitKey(0)

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
rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

# Sharpen the rotated image
def sharpen_image(image):
    # Create a sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

sharpened = sharpen_image(rotated)

# Display the sharpened corrected image
cv2.imshow('Sharpened Corrected Image', sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()
