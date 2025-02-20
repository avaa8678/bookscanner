import cv2
import numpy as np
import os

#p_width=0
#p_height=0

def display_image(window_title, image, width, height):
    """Display the image in a window with a set size."""
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, width, height)
    cv2.imshow(window_title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

def find_document_contour(cnts, base_epsilon=0.02, max_iter=5):
    """
    Loops over contours and attempts to approximate a quadrilateral.
    If none are found with the base epsilon, it increases epsilon gradually.
    Returns the approximated quadrilateral (4 points) or None.
    """
    # First, try with the base epsilon factor
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, base_epsilon * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    
    # If not found, try increasing epsilon up to 2*base_epsilon
    for factor in np.linspace(base_epsilon, base_epsilon * 2, max_iter):
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, factor * peri, True)
            if len(approx) == 4:
                print(f"Quadrilateral found using epsilon factor: {factor}")
                return approx.reshape(4, 2)
    
    print("find_document_contour Document contour not found.")
    return None

def detect_document(image):
    """
    Detects the document contour in the image by:
      - Converting to grayscale and blurring
      - Detecting edges via Canny
      - Finding contours and approximating to a polygon
    Returns the four corner points if found, otherwise None.
    """
    # 1. Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    #filtered = cv2.bilateralFilter(gray, 20, 30, 30)

    # Optionally compute an Otsu threshold for reference (not used for Canny)
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Apply Canny edge detection with tuned thresholds
    edged = cv2.Canny(blurred, 50, 150)

    # 3. Apply morphological closing to fill in gaps in the edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # display_image('Blurred Image', blurred, p_width, p_height)
    # display_image('Threshold (Otsu)', thresh, p_width, p_height)
    # display_image('Canny Edges', edged, p_width, p_height)
    # display_image('Closed Edges', closed, p_width, p_height)

    # Find contours and sort by area (largest first)
    #cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
    #cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:3]

    #DEBUG print the countour
    for i, c in enumerate(cnts):
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        print(f"Contour #{i}: area = {area}, boundingRect = (x={x}, y={y}, w={w}, h={h})")

        # If you also want to visualize each contour:
        # Make a copy of the image so we don't overwrite previous contours
        temp = image.copy()
        cv2.drawContours(temp, [c], -1, (0,255,0), 3)

    #DEBUGGING draw the countour
    # for c in cnts:
    #     cv2.drawContours(image, [c], -1, (0, 255, 0), 3)
    # display_image('Contour Image', image, p_width, p_height)

    # Attempt to find a quadrilateral in the sorted contours
    docCnt = find_document_contour(cnts, base_epsilon=0.02, max_iter=5)
    if docCnt is None:
        print("detect document No document contour found. Exiting.")
        exit()    

    return docCnt
    # Loop over the contours to find a quadrilateral
    # for c in cnts:
    #     peri = cv2.arcLength(c, True)
    #     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    #     if len(approx) == 4:
    #         return approx.reshape(4, 2)
    #     print("Document contour not found.")
    # return None

def process_image(image, scale=0.3):
    """
    Processes the image by detecting the document and applying a perspective warp.
    Returns the warped image or None if the document was not detected.
    """
    height, width = image.shape[:2]
    p_width = int(width * scale)
    p_height = int(height * scale)

    # Optionally display the original image for reference.
    #display_image('Original Image', image, p_width, p_height)

    pts = detect_document(image)
    if pts is None:
        print("Document contour not found.")
        return None

    warped = four_point_transform(image, pts)
    #display_image('Warped Image', warped, p_width, p_height)
    return warped

if __name__ == "__main__":
    # For testing, process a single image.
    img_path = r'C:\Users\james\OneDrive\Documents\Coding\Bookscanner\bookscanner\examples\P1231231.jpg'
    image = cv2.imread(img_path)
    
    scale=0.3
    height, width = image.shape[:2]
    p_width = int(width * scale)
    p_height = int(height * scale)

    if image is None:
        print("Error loading image.")
        exit()

    warped = process_image(image)
    if warped is not None:
        save_result(warped, img_path)
