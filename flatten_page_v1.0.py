import cv2
import numpy as np

def flatten_page(image):
    def order_points(pts):
        """Order points: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def show_debug(title, img):
        """Display image in a fixed 800x600 window."""
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 800, 600)
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyWindow(title)

    # Convert to grayscale and blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    show_debug("Threshold", thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    show_debug("Morphological Close", morph)

    cnts, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Contours found:", len(cnts))
    if not cnts:
        return image

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    screen_cnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        print("Contour vertices:", len(approx))
        if len(approx) == 4:
            screen_cnt = approx
            break

    if screen_cnt is None:
        print("No 4-corner contour found.")
        return image

    debug_img = image.copy()
    cv2.drawContours(debug_img, [screen_cnt], -1, (0, 255, 0), 2)
    show_debug("Detected Contour", debug_img)

    rect = order_points(screen_cnt.reshape(4, 2))
    width = max(int(np.linalg.norm(rect[2] - rect[3])), int(np.linalg.norm(rect[1] - rect[0])))
    height = max(int(np.linalg.norm(rect[1] - rect[2])), int(np.linalg.norm(rect[0] - rect[3])))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped

