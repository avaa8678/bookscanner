import os
from datetime import datetime
import cv2

# Define the save path
save_path = "/home/pi/code/bookscanner/results"
os.makedirs(save_path, exist_ok=True)

# Initialize OpenCV VideoCapture with device path
cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)

if not cap.isOpened():
    print("Failed to open camera.")
    exit()

# Set camera resolution (if supported)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print("Press 'Space' to capture an image. Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Flip the frame if necessary
    frame = cv2.flip(frame, -1)  # Flip around both axes if needed

    # Display the frame
    cv2.imshow("Camera Feed", frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # Press "Space" to capture
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pic_{timestamp}.jpg"
        filepath = os.path.join(save_path, filename)

        # Save the image
        cv2.imwrite(filepath, frame)
        print(f"Image saved as {filepath}")

    elif key == ord('q'):  # Press "q" to quit
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
