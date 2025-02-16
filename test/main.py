import cv2

# Load an image from your system
img = cv2.imread(r'C:\Users\james\Downloads\PXL_20241101_014528833.jpg')  # replace with your image path
# Resize the image to fit the screen (e.g., width = 800 pixels, height = 600 pixels)
img = cv2.resize(img, (768, 1024))  # Adjust dimensions as needed

# # Display the image in a window
# cv2.imshow('Image', img)

# # Convert to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Grayscale Image', gray)

# # Apply Gaussian Blur
# blur = cv2.GaussianBlur(img, (5, 5), 0)
# cv2.imshow('Blurred Image', blur)

# #Detect edges using Canny Edge Detection
# edges = cv2.Canny(img, 100, 200)
# cv2.imshow('Edges', edges)

# # Draw a rectangle
# cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), 2)
# # Draw a circle
# cv2.circle(img, (300, 300), 50, (255, 0, 0), -1)
# # Draw a line
# cv2.line(img, (100, 100), (400, 400), (0, 0, 255), 3)
# Add text
# cv2.putText(img, 'Andy Chen', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
# cv2.imshow('Shapes on Image', img)

# # Get the image center
# center = (img.shape[1] // 2, img.shape[0] // 2)

# # Create a rotation matrix
# M = cv2.getRotationMatrix2D(center, 45, 1.0)

# # Perform the rotation
# rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
# cv2.imshow('Rotated Image', rotated)

# # Flip the image (0 = vertical, 1 = horizontal, -1 = both axes)
# flipped = cv2.flip(img, 1)
# cv2.imshow('Flipped Image', flipped)

# Wait indefinitely for any keypress
cv2.waitKey(0)
# Close the window after keypress
cv2.destroyAllWindows()
