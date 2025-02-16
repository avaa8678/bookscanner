import cv2
import numpy as np

# Specify the image path
img_path = r'C:\Users\james\OneDrive\Documents\Coding\Bookscanner\bookscanner\examples\p1.jpg'

# Load the image
image = cv2.imread(img_path)

# Check if image was loaded successfully
if image is None:
    print("Error loading image")
    exit()

# Define the source points (corners of the region to be transformed)
# These points should be chosen in the order: top-left, top-right, bottom-right, bottom-left.
# Replace these values with the coordinates corresponding to your specific image.
src_pts = np.float32([
    [100, 50],    # top-left corner
    [400, 50],    # top-right corner
    [400, 300],   # bottom-right corner
    [100, 300]    # bottom-left corner
])

# Define the width and height for the output (destination image)
width, height = 300, 250  # adjust these dimensions as needed

# Define the destination points for a "birds-eye view"
dst_pts = np.float32([
    [0, 0],           # top-left corner in destination image
    [width - 1, 0],   # top-right corner
    [width - 1, height - 1],  # bottom-right corner
    [0, height - 1]   # bottom-left corner
])

# Compute the perspective transform matrix
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# Apply the perspective transformation
warped = cv2.warpPerspective(image, M, (width, height))

# Display the original and warped images
cv2.imshow("Original Image", image)
cv2.imshow("Warped Image", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
