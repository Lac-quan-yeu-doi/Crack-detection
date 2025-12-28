import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import random

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

def inpaint_crack_direction_aware(original_img, mask, 
                                  patch_height=17,   # perpendicular to crack (covers thickness)
                                  patch_width=51,    # along the crack direction
                                  search_range=80,   # how far to search up/down
                                  num_candidates=15, # how many best patches to blend
                                  step_sampling=5):  # speed up by processing every Nth crack pixel
    """
    Your original idea: Use an elongated patch that scans perpendicular to the crack
    to find the best matching healthy region and blend it in.
    Optimized for mostly horizontal cracks (common in roads).
    """
    img = original_img.copy().astype(np.float32)
    h, w = img.shape[:2]
    
    # Handle grayscale
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).astype(np.float32)
        was_grayscale = True
    else:
        was_grayscale = False

    mask_bool = (mask > 127)
    if mask_bool.ndim == 3:
        mask_bool = mask_bool[:, :, 0]

    # Get list of crack pixels
    crack_ys, crack_xs = np.where(mask_bool)
    if len(crack_ys) == 0:
        print("No crack pixels found.")
        return original_img.astype(np.uint8)

    half_h = patch_height // 2
    half_w = patch_width // 2

    # Pad image and mask to handle borders safely
    pad_y = half_h + search_range
    pad_x = half_w + search_range
    img_padded = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REFLECT)
    
    mask_padded = cv2.copyMakeBorder(mask_bool.astype(np.uint8), pad_y, pad_y, pad_x, pad_x,
                                     cv2.BORDER_CONSTANT, value=0)
    mask_padded = mask_padded > 0  # boolean

    result = img.copy()

    print(f"Processing {len(crack_ys)} crack pixels (sampling every {step_sampling} for speed)...")

    # Process crack pixels (with sampling for speed)
    for i in range(0, len(crack_ys), step_sampling):
        py, px = crack_ys[i], crack_xs[i]

        py_pad = py + pad_y
        px_pad = px + pad_x

        # Extract target patch (contains part of the crack)
        target_slice_y = slice(py_pad - half_h, py_pad + half_h + 1)
        target_slice_x = slice(px_pad - half_w, px_pad + half_w + 1)
        target_patch = img_padded[target_slice_y, target_slice_x]

        # Local mask for this patch
        local_mask = mask_padded[target_slice_y, target_slice_x]
        valid_mask = ~local_mask  # known healthy pixels

        if not np.any(valid_mask):
            continue

        # Search perpendicular (up and down) for best matching patches
        candidates = []
        for dy in range(-search_range, search_range + 1, 8):  # step 8 for speed
            if dy == 0:
                continue
            sy_pad = py_pad + dy

            source_slice_y = slice(sy_pad - half_h, sy_pad + half_h + 1)
            if source_slice_y.start < 0 or source_slice_y.stop > img_padded.shape[0]:
                continue

            source_patch = img_padded[source_slice_y, target_slice_x]

            # SSD error only on known (valid) pixels
            diff = (source_patch - target_patch) ** 2
            error = np.sum(diff[valid_mask]) / (np.sum(valid_mask) + 1e-8)

            candidates.append((error, source_patch, dy))

        if not candidates:
            continue

        # Sort and take top candidates
        candidates.sort(key=lambda x: x[0])
        top_candidates = candidates[:num_candidates]

        # Weighted blend of best matches
        blended = np.zeros_like(target_patch)
        total_weight = 0.0
        for error, source_patch, _ in top_candidates:
            weight = 1.0 / (error + 1e-6)
            blended += source_patch * weight
            total_weight += weight

        if total_weight > 0:
            blended /= total_weight

        # Apply only to crack region in this patch
        result_slice_y = slice(py - half_h, py + half_h + 1)
        result_slice_x = slice(px - half_w, px + half_w + 1)
        fill_region = local_mask

        if result_slice_y.start >= 0 and result_slice_y.stop <= h and \
           result_slice_x.start >= 0 and result_slice_x.stop <= w:
            result[result_slice_y, result_slice_x][fill_region] = blended[fill_region]

        if (i // step_sampling) % 50 == 0:
            print(f"  Progress: {i // step_sampling + 1}/{(len(crack_ys) - 1) // step_sampling + 1}")

    # Final clip and convert
    result = np.clip(result, 0, 255).astype(np.uint8)
    if was_grayscale:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    return result

# ==================== Full Testing Pipeline ====================
def test_your_crack_inpainting(image_path, output_dir="your_method_results"):
    os.makedirs(output_dir, exist_ok=True)

    original, mask = crack_detection(image_path, output_dir)  # Your function
    
    # Clean and dilate mask
    mask_dilated = mask.copy()

    cv2.imwrite(os.path.join(output_dir, "detected_mask.png"), mask_dilated)

    print("Running your direction-aware crack inpainting...")
    inpainted = inpaint_crack_direction_aware(
        original, mask_dilated,
        patch_height=17,
        patch_width=71,
        search_range=120,
        num_candidates=20,
        step_sampling=3
    )

    # Save result
    output_path = os.path.join(output_dir, "your_direction_aware_result.jpg")
    cv2.imwrite(output_path, inpainted)
    print(f"Result saved to: {output_path}")

    # Visualize
    plt.figure(figsize=(15, 8))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Detected Crack Mask")
    plt.imshow(mask_dilated, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Your Method:\nDirection-Aware Patch Blending")
    plt.imshow(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return inpainted


if __name__ == "__main__":
    image_path = "example/065.jpg"  # Update if needed
    test_your_crack_inpainting(image_path)

