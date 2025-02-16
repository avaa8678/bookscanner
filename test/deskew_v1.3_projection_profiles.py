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

# Morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)

# Define the center of the image for rotation
(h, w) = morph.shape[:2]
center = (w // 2, h // 2)

# Define a range of angles to rotate and evaluate
angle_range = np.arange(-50, 50.1, 0.1)  # Adjusted range based on expected skew
variances = []

# Rotate the image by each angle and compute the variance of the horizontal projection profile
for angle in angle_range:
    # Rotate the image
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(morph, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    # Crop the rotated image to the original size
    # For this case, the rotated image is the same size, so we can skip cropping
    # Optionally, you can define a ROI (Region of Interest) if needed

    # Compute the horizontal projection profile (sum of pixel values along each row)
    projection = np.sum(rotated, axis=1)

    # Calculate the variance of the projection profile
    variance = np.var(projection)
    variances.append(variance)

# Find the angle with the highest variance (best alignment of horizontal lines)
best_angle = angle_range[np.argmax(variances)]
print(f"Best angle for correction: {best_angle} degrees")

# Rotate the original image by the best angle
M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

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

sharpened = unsharp_mask(rotated_image, kernel_size=(9, 9), sigma=2.0, amount=2.0)

# Display the sharpened corrected image
cv2.imshow('Sharpened Corrected Image', sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()
