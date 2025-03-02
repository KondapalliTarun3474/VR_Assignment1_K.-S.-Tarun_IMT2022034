import cv2
import numpy as np
import random

# Load and preprocess image
image = cv2.imread("input_images/IMG20250302134045_BURST002.jpg")
seg = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#applying histogram equivalization
# gray = cv2.equalizeHist(gray)
# cv2.imwrite("histo.jpg", gray)
blurred = cv2.GaussianBlur(gray, (5, 5), 2)
#cv2.imwrite("Gaussianblurred.jpg", blurred)
_, binary = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)    #making the image binary
cv2.imwrite("output_images/binary.jpg", binary)

#edge detection using canny edge detector
edges = cv2.Canny(binary, 50, 150)
#cv2.imwrite("edgescanny.jpg", edges)
kernel = np.ones((3,3), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=3)    # Apply dilation to strengthen edges and close small gaps
#cv2.imwrite("edgesdia.jpg", edges)

coin_count =0

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#print(f"Number of contours found: {len(contours)}")
# Store unique colors
#used_colors = set()

# def generate_bright_color():   #just a function that picks a random bright colour and ensures same colour isn't picked twice
#     while True:
#         color = tuple(np.random.randint(100, 256, 3).tolist())  # Ensure each channel is at least 100
#         if color not in used_colors:
#             used_colors.add(color)
#             return color

for  contour in contours:
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)

    if perimeter == 0:  # Avoid division by zero
        continue

    # Compute circularity
    circularity = (4 * np.pi * area) / (perimeter ** 2)

    # Check if contour is a valid coin(adjusted thresholds)
    if  area> 500 and 0.7 < circularity < 1.2:
        coin_count += 1
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 5)  # Draw outline

        cv2.drawContours(seg, [contour], -1, (random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)), -1)  # Fill segmented coin
        # color = generate_bright_color()  



# Display count on image
height, width = image.shape[:2]
cv2.putText(image, f"Total Coins: {coin_count}", (50, height - 50),
            cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 255), 8)

cv2.imwrite("output_images/detected_coins.jpg", image)
cv2.imwrite("output_images/edges.jpg", edges)
cv2.imwrite("output_images/segmented_coins.jpg", seg)


#print("Results saved successfully!")
cv2.destroyAllWindows()

print(f"Total Number of Coins: {coin_count}")
