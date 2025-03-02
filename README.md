# Visual Recognition Assignment 1

## Author: K. S. Tarun  
**Student ID:** IMT2022034  
**GitHub Repository:** [Insert Link Here]  

---

## Overview  
This project consists of two major tasks:  
1. **Coin Detection, Segmentation, and Counting**  
2. **Image Stitching for Panorama Generation**  

Both tasks utilize computer vision techniques implemented using OpenCV and related libraries.

---

## Part 1: Coin Detection, Segmentation, and Counting  

### Objective  
Detect, segment, and count coins from an image containing scattered Indian coins using computer vision techniques.

### Implementation Steps  
1. **Preprocessing the Image**  
   - Convert the image to grayscale.  
   - Apply Gaussian blur for noise reduction.  
2. **Edge Detection**  
   - Use Canny edge detection to find edges of coins.  
   - Apply dilation to strengthen edges.  
3. **Contour Detection and Filtering**  
   - Detect contours in the processed image.  
   - Filter using circularity and area thresholds to remove false positives.  
4. **Segmentation and Counting**  
   - Highlight detected coins with green contours.  
   - Assign random colors to segmented coins.  
   - Display and save the final output.  

### Challenges and Refinements  
- **Direct Canny Edge Detection** produced incomplete results due to lighting variations.  
- **Histogram Equalization** increased noise and background artifacts.  
- **Binary Thresholding** provided the best separation between coins and background.  
- **Refined Contour Selection** eliminated noise by adjusting circularity and area thresholds.  

### Final Approach  
- **Preprocessing:** Gaussian blur  
- **Binary Thresholding:** Better foreground-background separation  
- **Edge Detection:** Canny method  
- **Dilation:** Strengthen edges  
- **Contour Detection & Filtering:** Circularity & area-based selection  
- **Visualization:** Mark and count detected coins  

### Outcome  
Successfully detected and counted coins with high accuracy while minimizing noise impact.

---

## Part 2: Image Stitching for Panorama Generation  

### Objective  
Stitch three images into a single panorama using feature detection and homography transformation.

### Implementation Steps  
1. **Keypoint Detection and Feature Extraction**  
   - Convert images to grayscale.  
   - Detect key points using SIFT.  
2. **Feature Matching**  
   - Use BFMatcher with L2 norm for optimal SIFT feature matching.  
   - Apply cross-checking to ensure one-to-one matches.  
3. **Homography Estimation**  
   - Compute transformation matrix aligning images.  
   - Map corresponding key points between images.  
4. **Image Warping and Stitching**  
   - Warp images using the homography matrix.  
   - Overlay images to generate a seamless panorama.  
5. **Final Panorama Generation**  
   - Stitch image3 (right) to image1 (middle).  
   - Stitch image2 (left) to the intermediate result.  
   - Save the final panorama as `panorama_3.jpg`.  

### Outcome  
Successfully created a panoramic image by aligning and blending multiple images using computer vision techniques.

---

## Requirements  
- Python 3.x  
- OpenCV  
- NumPy  
- Matplotlib  

Install dependencies using:  
```bash
pip install opencv-python numpy matplotlib
```

---

## Running the Code  
- **Coin Detection:** Run `coin_detection.py`  
- **Image Stitching:** Run `image_stitching.py`  

Ensure the required images are placed in the appropriate directory.

---

## Conclusion  
This project demonstrates the use of computer vision techniques for object detection and image alignment. The final implementations successfully achieve:  
✔️ Accurate coin detection and segmentation  
✔️ Seamless image stitching for panorama generation  

---
