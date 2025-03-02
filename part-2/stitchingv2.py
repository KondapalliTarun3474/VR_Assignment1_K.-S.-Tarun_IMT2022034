import cv2
import numpy as np

def detect_keypoints(image):  # Function to detect keypoints and extract descriptors using SIFT
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)    # Detect keypoints and compute descriptors
    return keypoints, descriptors

def match_features(descriptors1, descriptors2):   # Function to match features between two sets of descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)    # Match descriptors
    matches = sorted(matches, key=lambda x: x.distance)   # Sort matches by distance (lower is better)
    return matches

def get_homography(matches, keypoints1, keypoints2):     # Function to compute homography matrix from matched features
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)   # Extract source points
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)   # Extract destination points

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H

def stitching(image1, image2):   # Function to stitch two images together
    keypoints1, descriptors1 = detect_keypoints(image1)
    keypoints2, descriptors2 = detect_keypoints(image2)
    # Draw keypoints on images and save them for visualization
    keypoint_image1 = cv2.drawKeypoints(image1, keypoints1, None, color=(0, 255, 0))
    keypoint_image2 = cv2.drawKeypoints(image2, keypoints2, None, color=(0, 255, 0))
    cv2.imwrite(f"output_images/KeypointsImage{i}1.jpg", keypoint_image1)
    cv2.imwrite(f"output_images/KeypointsImage{i}2.jpg", keypoint_image2)
    matches = match_features(descriptors1, descriptors2)   # Match features between the two images

     # Draw feature matches and save for visualization
    match_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], None, flags=2)
    cv2.imwrite(f"output_images/FeatureMatches{i}.jpg", match_image)
    H = get_homography(matches, keypoints1, keypoints2)   # Compute homography matrix
    height, width, _ = image2.shape    # Get dimensions of second image
    warped_image1 = cv2.warpPerspective(image1, H, (width * 2, height))   # Warp first image using homography
    warped_image1[0:height, 0:width] = image2 # Overlay second image onto warped image
    return warped_image1
    cv2.destroyAllWindows()

i=1 # Counter for saving intermediate images
# Load input images
image1 = cv2.imread("input_images/middle.jpg")  # Load middle image
image2 = cv2.imread("input_images/left.jpg")  # Load left image
image3 = cv2.imread("input_images/right.jpg")  # Load right image

half_stitched = stitching (image3, image1)  # Stitch right and middle images first
cv2.imwrite("output_images/half_stitched.jpg", half_stitched)    # Save intermediate stitched result
i=2 # Update counter for second stitching step
# Stitch the left image with the already stitched image
full_stitched = stitching(half_stitched, image2)
cv2.imwrite("output_images/panorama_3.jpg", full_stitched)   # Save final panorama image
cv2.destroyAllWindows()

# # Detect keypoints & descriptors
# keypoints1, descriptors1 = detect_keypoints(image1)
# keypoints2, descriptors2 = detect_keypoints(image2)
# keypoints3, descriptors3 = detect_keypoints(image3)

# Match features between left & middle
#matchesrm = match_features(descriptors3, descriptors1)


#matchesml = match_features(descriptors1, descriptors3)


# Get homographies
#H = get_homography(matchesrm, keypoints3, keypoints1)
#H13 = get_homography(matches13, keypoints1, keypoints3)

# Warp left and right images to align with middle
#height, width, _ = image1.shape
# canvas_width = width * 3  # Create a large enough canvas for all three images
# canvas_height = height

# warped_left = cv2.warpPerspective(image2, H12, (canvas_width, canvas_height))
# warped_right = cv2.warpPerspective(image3, H13, (canvas_width, canvas_height))

# # Place middle image on the panorama
# panorama = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
# panorama[0:height, width:2*width] = image1  # Center image
# panorama[0:height, :width] = warped_left[:, :width]  # Left image
# panorama[0:height, 2*width:] = warped_right[:, 2*width:]  # Right image
#warped_image1 = cv2.warpPerspective(image3, H, (width * 2, height))

# Place image2 on the warped image
#warped_image1[0:height, 0:width] = image3
#cv2.imwrite("panorama.jpg", warped_image1)


#cv2.imwrite("panorama_3.jpg", panorama)
#cv2.destroyAllWindows()
