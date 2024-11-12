import cv2
from picamera2 import Picamera2

# Initialize Picamera2
picam2 = Picamera2()

# Set up preview configuration
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)

# Set white balance mode to automatic
picam2.set_controls({"AwbEnable": True, "AwbMode": 1})  # Mode 1 is typically automatic
#picam2.set_controls({"ColourGains": (1.5, 1.8)})  # Adjust these values as needed


# Start the camera
picam2.start()

print("Press 'q' to exit.")

# Display the camera feed
while True:
    frame = picam2.capture_array()
    cv2.imshow("Picamera2 Video Feed", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the display window and stop the camera
cv2.destroyAllWindows()
picam2.stop()
