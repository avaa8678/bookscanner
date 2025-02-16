import cv2
import numpy as np

def flatten_page_vertical(image, num_slices=50, debug=False):
    """
    Flatten a page by slicing vertically and applying a small perspective warp per slice.
    :param image: Input BGR image (NumPy array).
    :param num_slices: Number of vertical slices to use.
    :param debug: If True, show intermediate images.
    :return: Flattened BGR image.
    """
    h, w = image.shape[:2]
    slice_width = w // num_slices

    # Store warped slices for final horizontal concatenation
    warped_slices = []

    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # (Optional) Denoise or do morphological ops if needed
    # gray = cv2.medianBlur(gray, 3)

    for i in range(num_slices):
        # Define the horizontal range for this slice
        x_start = i * slice_width
        x_end = (i+1)*slice_width if i < num_slices-1 else w

        # Extract the slice from the original color image
        slice_color = image[:, x_start:x_end]

        # For edge detection, use the grayscale slice
        slice_gray = gray[:, x_start:x_end]

        # Detect edges in this vertical slice
        edges = cv2.Canny(slice_gray, 50, 150)
        coords = np.column_stack(np.where(edges > 0))

        if len(coords) == 0:
            # If no edges found, just keep the slice as is
            warped_slices.append(slice_color)
            continue

        # y_min, y_max among the edges in this slice
        y_min = np.min(coords[:, 0])
        y_max = np.max(coords[:, 0])

        # Construct source points (approx corners of the region)
        # We want to "stretch" the slice from top to bottom
        src_pts = np.float32([
            [0,        y_min],
            [slice_color.shape[1], y_min],
            [slice_color.shape[1], y_max],
            [0,        y_max]
        ])
        # Destination points: the full height of the slice
        dst_pts = np.float32([
            [0,               0],
            [slice_color.shape[1], 0],
            [slice_color.shape[1], slice_color.shape[0]],
            [0,               slice_color.shape[0]]
        ])

        # Compute perspective transform
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_slice = cv2.warpPerspective(slice_color, M,
                                           (slice_color.shape[1], slice_color.shape[0]))

        warped_slices.append(warped_slice)

        if debug:
            debug_img = slice_color.copy()
            # Draw lines where top/bottom edges are found
            cv2.line(debug_img, (0, y_min), (debug_img.shape[1], y_min), (0,255,0), 2)
            cv2.line(debug_img, (0, y_max), (debug_img.shape[1], y_max), (0,0,255), 2)
            cv2.imshow(f"Slice {i} original", debug_img)
            cv2.imshow(f"Slice {i} warped", warped_slice)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # Concatenate all warped slices horizontally
    flattened = np.concatenate(warped_slices, axis=1)
    return flattened

def main():
    # Load the sample image
    image_path = r"C:\Users\james\OneDrive\Documents\Coding\Bookscanner\bookscanner\examples\f1.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print("Could not load image.")
        return

    # Flatten the page vertically
    flattened_image = flatten_page_vertical(image, num_slices=5, debug=False)

    # Save and display the result
    out_path = image_path.replace(".jpg", "_flattened_vertical.jpg")
    cv2.imwrite(out_path, flattened_image)
    print(f"Flattened image saved to {out_path}")

    cv2.imshow("Flattened Image", flattened_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
