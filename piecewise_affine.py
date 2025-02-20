import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform, io

def detect_text_lines(binary_image):
    """
    Very simplistic text line detection using horizontal projection.
    Returns a list of (y_start, y_end) tuples for each text line.
    In practice, you'd want something more robust (contours, OCR-based, etc.).
    """
    # Sum along each row
    row_sum = np.sum(binary_image, axis=1)
    threshold = np.max(row_sum) * 0.1  # pick a fraction of max as threshold
    lines = []
    in_line = False
    start = 0

    for i, val in enumerate(row_sum):
        if val > threshold and not in_line:
            in_line = True
            start = i
        elif val <= threshold and in_line:
            in_line = False
            end = i
            lines.append((start, end))

    # If we ended in a line
    if in_line:
        lines.append((start, len(row_sum)-1))

    return lines

def find_line_baseline(binary_line_slice):
    """
    Estimate a 'baseline' for a given text line slice.
    A trivial approach: find the lowest (or highest) white pixel in each column.
    In reality, you'd do morphological analysis or connected components, etc.
    Returns a list of (x, y) points for the baseline.
    """
    h, w = binary_line_slice.shape
    baseline_points = []
    for x in range(w):
        col = binary_line_slice[:, x]
        # indices of white pixels in this column
        white_pixels = np.where(col > 0)[0]
        if len(white_pixels) > 0:
            # pick the bottom-most white pixel
            y = white_pixels[-1]
            baseline_points.append((x, y))
        else:
            # no white pixel found, just guess middle
            baseline_points.append((x, h // 2))
    return baseline_points

def build_piecewise_transform(src_points, dst_points, image_shape):
    """
    Build a piecewise affine transform using skimage.
    src_points, dst_points: Nx2 arrays of matched coordinates.
    image_shape: (height, width)
    Returns a warp function or transform you can apply to the entire image.
    """
    tform = transform.PiecewiseAffineTransform()
    tform.estimate(np.array(src_points), np.array(dst_points))
    # We'll create a warped image of the same shape
    warped = transform.warp(
        image_shape, 
        inverse_map=tform.inverse,
        output_shape=image_shape.shape  # same shape
    )
    return tform

def warp_entire_line(image, line_bounds, baseline_points):
    """
    Warps the region of the image defined by line_bounds so that the baseline 
    (given by baseline_points) is mapped to a nearly horizontal line.
    A small jitter is added to the destination points to avoid collinearity.
    """
    y_start, y_end = line_bounds
    line_slice = image[y_start:y_end, :]  # slice out the line region
    h, w = line_slice.shape[:2]

    # Build source points from baseline (in slice coordinates)
    src_points = []
    for (x, y) in baseline_points:
        src_points.append([x, y])

    # Build destination points: map each x to a nearly constant y, 
    # but add a small jitter to avoid perfectly collinear points.
    base_line_y = int(0.9 * h)
    dst_points = []
    for (x, y) in baseline_points:
        jitter = np.random.uniform(-1, 1)  # adjust the range as needed
        new_y = base_line_y + jitter
        dst_points.append([x, new_y])

    # Convert the slice to float for scikit-image (range [0,1])
    line_slice_float = line_slice.astype(np.float32) / 255.0

    # Build and apply the piecewise affine transform
    tform = transform.PiecewiseAffineTransform()
    try:
        tform.estimate(np.array(src_points), np.array(dst_points))
    except Exception as e:
        print("Error estimating transform (likely due to degenerate points):", e)
        return image  # Return the original image if warping fails

    warped_slice = transform.warp(
        line_slice_float, tform, output_shape=(h, w)
    )
    warped_slice_uint8 = (warped_slice * 255).astype(np.uint8)

    # Put the warped slice back into the image.
    result = image.copy()
    # Here we assume the warped slice is grayscale; if not, adjust as needed.
    result[y_start:y_end, :] = warped_slice_uint8
    return result

def line_based_flatten(image_path):
    # Load image in grayscale
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print("Error loading image.")
        return None

    # Binarize (simple threshold or Otsu)
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Detect text lines by horizontal projection (simple example)
    lines = detect_text_lines(binary)

    # Convert original image to color so we can overlay warps
    original_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # For each line, find the baseline and warp it
    flattened = original_bgr.copy()
    for (y_start, y_end) in lines:
        # Get the line slice in the binary image
        line_slice_binary = binary[y_start:y_end, :]
        baseline_pts = find_line_baseline(line_slice_binary)
        # Warp that line so baseline is horizontal
        flattened = warp_entire_line(flattened, (y_start, y_end), baseline_pts)

    return flattened

def main():
    image_path = r"C:\Users\james\OneDrive\Documents\Coding\Bookscanner\bookscanner\examples\f1.jpg"
    flattened = line_based_flatten(image_path)
    if flattened is None:
        return

    # Save and display
    out_path = image_path.replace(".jpg", "_line_flattened.jpg")
    cv2.imwrite(out_path, flattened)
    print(f"Flattened image saved to {out_path}")

    cv2.imshow("Line-based Flattened", flattened)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
