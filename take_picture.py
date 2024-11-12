import os
from datetime import datetime
from picamera2 import Picamera2
from libcamera import Transform
import cv2

# Define the save path
save_path = "/home/pi/code/bookscanner/results"
os.makedirs(save_path, exist_ok=True)

# Set desired preview width and height for display only
p_width = 1600
p_height = 1200

# Initialize the camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (2592, 1944), "format": "RGB888"},  # Optimized for smooth preview
    transform=Transform(hflip=0, vflip=0)
)
picam2.configure(config)

# Optimize camera controls for OV5647
picam2.set_controls({
    "AwbEnable": True,                # Enable auto white balance
    "AwbMode": 1,                     # Use automatic white balance mode
    "ExposureTime": 20000,            # Set fixed exposure time in microseconds
    "AnalogueGain": 1.0,              # Set fixed gain for stable exposure
    "Sharpness": 1.5,                 # Increase sharpness for clearer image
    "Contrast": 1.2,                  # Enhance contrast slightly
    "Saturation": 1.0,                # Neutral saturation
    "Brightness": 0.05,                # Set neutral brightness
    "AeEnable": True                  # Enable auto-exposure
})

# Start the camera
picam2.start()

# Create a named OpenCV window and resize it for display
cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera Feed", p_width, p_height)

print("Press 'Space' to capture an image. Press 'q' to quit.")

while True:
    # Capture frame-by-frame at full resolution
    frame = picam2.capture_array()

    # Resize the frame to fit the preview window dimensions
    preview_frame = cv2.resize(frame, (p_width, p_height))

    # Display the resized frame
    cv2.imshow("Camera Feed", preview_frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # Press "space" to capture
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pic_{timestamp}.jpg"
        filepath = os.path.join(save_path, filename)

        # Capture and save the full-resolution image
        picam2.capture_file(filepath)
        print(f"Image saved as {filepath}")

    elif key == ord('q'):  # Press "q" to quit
        break

# Close windows and stop the camera
cv2.destroyAllWindows()
picam2.stop()
