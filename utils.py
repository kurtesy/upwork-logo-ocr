import cv2
import numpy as np

def _calculate_similarity_from_arrays(img1_array: np.ndarray, img2_array: np.ndarray) -> dict | None:
    """
    Core logic to calculate similarity between two images provided as NumPy arrays.
    
    Args:
        img1_array (np.ndarray): First image as a NumPy array.
        img2_array (np.ndarray): Second image as a NumPy array.

    Returns:
        dict: A dictionary containing color similarity, shape similarity,
              and a combined similarity score.
    """
    try:
        if img1_array is None or img1_array.size == 0 or \
           img2_array is None or img2_array.size == 0:
            print("Error: One or both image arrays are invalid.")
            return None

        # --- 1. Color Similarity (using HSV histograms) ---
        # Ensure images are 3-channel for HSV conversion if they are grayscale
        img1_for_color = cv2.cvtColor(img1_array, cv2.COLOR_GRAY2BGR) if len(img1_array.shape) == 2 else img1_array
        img2_for_color = cv2.cvtColor(img2_array, cv2.COLOR_GRAY2BGR) if len(img2_array.shape) == 2 else img2_array

        img1_hsv = cv2.cvtColor(img1_for_color, cv2.COLOR_BGR2HSV)
        img2_hsv = cv2.cvtColor(img2_for_color, cv2.COLOR_BGR2HSV)

        # Calculate histograms for the H, S channels
        hist_img1 = cv2.calcHist([img1_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist_img2 = cv2.calcHist([img2_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

        cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        color_similarity = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)

        # --- 2. Shape Similarity (using contours and cv2.matchShapes) ---
        # If images are already grayscale, use them directly. Otherwise, convert.
        img1_gray = img1_array if len(img1_array.shape) == 2 else cv2.cvtColor(img1_array, cv2.COLOR_BGR2GRAY)
        img2_gray = img2_array if len(img2_array.shape) == 2 else cv2.cvtColor(img2_array, cv2.COLOR_BGR2GRAY)

        # Apply thresholding (simple binary threshold)
        _, thresh1 = cv2.threshold(img1_gray, 127, 255, cv2.THRESH_BINARY_INV)
        _, thresh2 = cv2.threshold(img2_gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        # Use cv2.RETR_EXTERNAL to get only external contours
        # Use cv2.CHAIN_APPROX_SIMPLE to compress segments
        contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        shape_similarity = 0  # Default to 0 if no contours found or comparison fails

        if contours1 and contours2:
            contour1_largest = max(contours1, key=cv2.contourArea)
            contour2_largest = max(contours2, key=cv2.contourArea)
            match_value = cv2.matchShapes(contour1_largest, contour2_largest, cv2.CONTOURS_MATCH_I1, 0.0)
            shape_similarity = 1.0 - match_value
            shape_similarity = max(0, shape_similarity)


        # --- 3. Combine Scores (Example: weighted average) ---
        # Adjust weights as needed
        weight_color = 0.6
        weight_shape = 0.4
        combined_similarity = (weight_color * color_similarity) + (weight_shape * shape_similarity)

        return {
            "color_similarity": color_similarity,
            "shape_similarity": shape_similarity,
            "combined_similarity": combined_similarity
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def calculate_image_similarity(image_source1, image_source2):
    """
    Calculates similarity between two images based on color histograms and shape contours.
    The sources can be file paths (str) or image arrays (np.ndarray).

    Args:
        image_source1 (str | np.ndarray): Path to the first image or the image as a NumPy array.
        image_source2 (str | np.ndarray): Path to the second image or the image as a NumPy array.

    Returns:
        dict: A dictionary containing color similarity, shape similarity,
              and a combined similarity score, or None if an error occurs.
    """
    img1_array = None
    img2_array = None

    if isinstance(image_source1, str):
        img1_array = cv2.imread(image_source1, cv2.IMREAD_UNCHANGED) # Read as is (color or grayscale)
    elif isinstance(image_source1, np.ndarray):
        img1_array = image_source1
    else:
        raise ValueError("image_source1 must be a file path (str) or a NumPy array.")

    if isinstance(image_source2, str):
        img2_array = cv2.imread(image_source2, cv2.IMREAD_UNCHANGED)
    elif isinstance(image_source2, np.ndarray):
        img2_array = image_source2
    else:
        raise ValueError("image_source2 must be a file path (str) or a NumPy array.")

    return _calculate_similarity_from_arrays(img1_array, img2_array)