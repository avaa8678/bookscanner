import cv2
import numpy as np
import os

def order_points(pts):
    """
    Orders a list of 4 points in the order:
    top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    # sum of coordinates: smallest is top-left, largest is bottom-right
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # difference: smallest is top-right, largest is bottom-left
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """
    Applies a perspective transform to the image using the 4 given points.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the width of the new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # Destination coordinates for "birds-eye view"
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

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

def detect_document(image):
    """
    Detects the document contour in the image by:
      - Converting to grayscale and blurring
      - Detecting edges via Canny
      - Finding contours and approximating to a polygon
    Returns the four corner points if found, otherwise None.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #edged = cv2.Canny(blurred, 75, 200)
    edged = cv2.Canny(blurred, 1, 10)


    # Find contours and sort by area (largest first)
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    # Loop over the contours to find a quadrilateral
    for c in cnts:
        peri = cv2.arcLength(c, True)
        #approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    print("Document contour not found.")
    return None

if __name__ == "__main__":
    # Provide the path to your image
    img_path = r'C:\Users\james\OneDrive\Documents\Coding\Bookscanner\bookscanner\examples\p1.jpg'
    image = cv2.imread(img_path)
    if image is None:
        print("Error loading image.")
        exit()

    pts = detect_document(image)
    if pts is None:
        exit()

    warped = four_point_transform(image, pts)

    # Display the original and scanned images (optional)
    cv2.imshow("Original Image", image)
    cv2.imshow("Scanned Image", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the result using your save_result function
    save_result(warped, img_path)
