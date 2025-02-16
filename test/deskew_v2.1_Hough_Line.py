import os
import cv2
import numpy as np
from flatten_page import flatten_page

def load_image(path):
    """Load the image from disk."""
    return cv2.imread(path)

def preprocess_image(image):
    """Convert to grayscale, equalize histogram, apply blurs, and threshold."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    gray_blurred = cv2.medianBlur(gray_blurred, 3)
    _, thresh = cv2.threshold(gray_blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def apply_morphology(thresh):
    """Apply morphological operations to the thresholded image."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
    return morph

def detect_edges(morph):
    """Detect edges using the Canny algorithm."""
    return cv2.Canny(morph, 30, 200, apertureSize=3)

def detect_lines(edges):
    """Detect lines using the Hough Line Transform."""
    return cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=20)

def compute_median_angle(lines):
    """Compute the median angle of valid detected lines."""
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
    return np.median(angles)

def deskew_image(image, median_angle):
    """Rotate the image to correct the skew."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return rotated

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0):
    """Sharpen the image using Unsharp Masking."""
    img_float = image.astype(np.float32)
    blurred = cv2.GaussianBlur(img_float, kernel_size, sigma)
    high_freq = img_float - blurred
    if threshold > 0:
        low_contrast_mask = np.absolute(high_freq) < threshold
        high_freq[low_contrast_mask] = 0
    sharpened = img_float + amount * high_freq
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def display_image(window_title, image, width, height):
    """Display the image in a window with a set size."""
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, width, height)
    cv2.imshow(window_title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_result(image, original_path):
    """
    Save the result image in a "/bookscanner/results" folder using the same file name as the original.
    """
    filename = os.path.basename(original_path)
    results_folder = os.path.join("bookscanner", "results")
    os.makedirs(results_folder, exist_ok=True)
    save_path = os.path.join(results_folder, filename)
    cv2.imwrite(save_path, image)
    print("Result saved as:", save_path)

def main():
    image_path = r'C:\Users\james\OneDrive\Documents\Coding\Bookscanner\bookscanner\examples\p6.jpg'
    # Load image
    image = load_image(image_path)

    SCALE = 0.3
    # Get the dimensions
    height, width = image.shape[:2]  # Extract only height and width, ignore channels
    # Scale the preview width and height
    p_width = int(width * SCALE)  # Convert to integer
    p_height = int(height * SCALE)  # Convert to integer

    # Preprocess image (grayscale, equalize, blur, threshold)
    thresh = preprocess_image(image)
    display_image('Threshold (Otsu)', thresh, p_width, p_height)
    
    # Morphological operations
    morph = apply_morphology(thresh)
    
    # Edge detection
    edges = detect_edges(morph)
    display_image('Canny Edges', edges, p_width, p_height)    
    
    # Hough Line Transform
    lines = detect_lines(edges)

    # DEBUGGING Visualize detected lines
    #
    line_img = image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length > 100:  # Minimum line length threshold
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    display_image('Detected Lines', line_img, p_width, p_height)        
    
    # Compute the median angle from valid lines
    median_angle = compute_median_angle(lines)
    
    # Deskew image based on median angle
    rotated = deskew_image(image, median_angle)
    
    # Sharpen the rotated image using Unsharp Masking
    # Ava - without shapen we can remove the "halo" around hte word.
    #sharpened = unsharp_mask(rotated, kernel_size=(9, 9), sigma=2.0, amount=2.0)


    flatten_img = flatten_page(rotated)

    result_img = flatten_img
    # Save the result image in the "../Bookscanner/bookscanner/results" folder
    save_result(result_img, image_path)
    # Display the final result
    display_image('Result Image', result_img, p_width, p_height)

if __name__ == "__main__":
    main()
