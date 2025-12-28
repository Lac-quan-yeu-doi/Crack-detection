import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import random

# ------------------- Approach 1: Morphological + Guided Inpainting -------------------
def inpaint_morph_guided(original_img, mask, kernel_size=15, iterations=3):
    """
    Close cracks morphologically, then use distance-guided inpainting.
    Fast and good for thin-to-medium cracks.
    """
    # Morphological closing to connect broken parts (optional)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    # Compute distance transform for better guidance
    dist = cv2.distanceTransform(255 - closed_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dist = np.clip(dist, 0, 20)  # Limit influence radius
    dist = (dist / dist.max() * 255).astype(np.uint8)

    # Use Telea (fast marching) - often better for thin structures
    inpainted = cv2.inpaint(
        original_img, closed_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA
    )

    return inpainted


# ------------------- Approach 2: Non-Local Means Texture Propagation -------------------
def inpaint_nlmeans(original_img, mask, strength=15, iterations=20):
    """
    Iteratively apply non-local means denoising targeted to crack regions.
    Preserves texture better than simple blur.
    """
    result = original_img.copy()
    mask_bool = mask > 127

    for i in range(iterations):
        # Apply NLM denoising (stronger in masked areas)
        denoised = cv2.fastNlMeansDenoisingColored(
            result,
            None,
            h=strength,
            hColor=strength,
            templateWindowSize=7,
            searchWindowSize=21,
        )
        # Copy denoised values only into crack regions
        result[mask_bool] = denoised[mask_bool]

        if i % 5 == 0:
            print(f"NLM iteration {i+1}/{iterations}")

    return result


# ------------------- Approach 3: Hybrid Median + Gaussian Filling -------------------
def inpaint_hybrid_median_gaussian(
    original_img, mask, median_ksize=9, gauss_ksize=5, iterations=50
):
    """
    Very fast targeted filling using median (edge-preserving) + Gaussian.
    Good compromise between speed and texture.
    """
    result = original_img.copy().astype(np.float32)
    mask_bool = mask > 127

    for i in range(iterations):
        # Median filter preserves edges/texture
        median = cv2.medianBlur(result.astype(np.uint8), median_ksize)
        # Gaussian for smooth blending
        blurred = cv2.GaussianBlur(median, (gauss_ksize, gauss_ksize), 0)

        # Propagate into crack regions
        result[mask_bool] = blurred[mask_bool]

    return np.clip(result, 0, 255).astype(np.uint8)


def crack_detection(image_path, output_dir="result"):
    os.makedirs(output_dir, exist_ok=True)

    original_color = cv2.imread(image_path)
    if original_color is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    gray_norm = gray.astype(np.float32) / 255.0

    # Gaussian blur and enhancement
    sigma = 11
    blur = cv2.GaussianBlur(gray_norm, (2 * math.ceil(2 * sigma) + 1,) * 2, sigma)
    enhanced = cv2.subtract(gray_norm, blur)

    # Histogram clipping
    high = np.percentile(enhanced, 50)
    enhanced = np.clip(enhanced, None, high)
    enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-8)

    # Sobel
    sobel_ksize = 9
    sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    mag = np.hypot(sobelx, sobely)
    ang = np.arctan2(sobely, sobelx)

    threshold = 3 * np.mean(mag)
    mag[mag < threshold] = 0

    # Oriented Non-Maximum Suppression (your implementation)
    def non_max_suppression(data, win):
        data_max = ndimage.maximum_filter(data, footprint=win, mode="constant")
        data_max[data != data_max] = 0
        return data_max

    def orientated_non_max_suppression(mag, ang):
        ang_quant = np.round(ang / (np.pi / 4)) % 4
        wins = [
            np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]),  # E
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # SE
            np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]),  # S
            np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),  # SW
        ]
        result = np.zeros_like(mag)
        for i, win in enumerate(wins):
            suppressed = non_max_suppression(mag, win)
            result[ang_quant == i] = suppressed[ang_quant == i]
        return result

    mag_nms = orientated_non_max_suppression(mag, ang)

    high_thresh = 0.5 * mag_nms.max()
    low_thresh = 0.2 * mag_nms.max()

    edges = np.zeros_like(mag_nms, dtype=np.uint8)
    edges[mag_nms > high_thresh] = 255
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    edges[mag_nms > low_thresh] = np.where(edges[mag_nms > low_thresh] > 0, 255, 0)

    # Morphological operations
    close_ksize = 25
    open_ksize = 6
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_ksize, close_ksize)
    )
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))

    binary_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)
    final_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)

    cv2.imwrite(os.path.join(output_dir, "binary_crack_map.png"), final_mask)

    return original_color, final_mask

if __name__ == "__main__":
    image_path = "example/001.jpg"  # Replace with your image path
    output_dir = "result"

    print("Starting crack detection...")
    original_color, crack_mask = crack_detection(image_path, output_dir)

    # Example: Try morphological guided (fastest, good quality)
    inpainted = inpaint_morph_guided(original_color, crack_mask, kernel_size=21, iterations=2)
    cv2.imwrite(os.path.join(output_dir, "crack_removed_morph_guided.jpg"), inpainted)

    # Or non-local means (best texture preservation among fast methods)
    inpainted = inpaint_nlmeans(original_color, crack_mask, strength=20, iterations=30)
    cv2.imwrite(os.path.join(output_dir, "crack_removed_nlmeans.jpg"), inpainted)
    
    # Or hybrid median (very fast)
    inpainted = inpaint_hybrid_median_gaussian(original_color, crack_mask, iterations=80)
    cv2.imwrite(os.path.join(output_dir, "crack_removed_hybrid_median.jpg"), inpainted)