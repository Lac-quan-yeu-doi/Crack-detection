# get median of growing square + weighted blend from 8-neighbors + final Gaussian

import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

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

    # Oriented Non-Maximum Suppression
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

def inpaint_median_weighted_blend(original_img, mask,
                                  step=6,
                                  ratio=0.05,
                                  max_size=181,
                                  neighbor_weight=0.6,     # Total weight for 8-neighbors (1 - this = median weight)
                                  gaussian_ksize=9,
                                  gaussian_sigma=1.2):
    """
    Your LATEST BEST idea:
    - Grow square to collect known (non-crack) pixels
    - Use MEDIAN of those pixels as main texture value
    - Blend with 8-connected known neighbors (including previously filled)
    - neighbor_weight = total weight shared among neighbors
    - Final targeted Gaussian on original crack areas
    """
    img = original_img.copy().astype(np.float32)
    h, w = img.shape[:2]

    if img.ndim == 2:
        img = img[..., np.newaxis]
        was_grayscale = True
        channels = 1
    else:
        was_grayscale = False
        channels = img.shape[2]

    mask_bool = (mask > 127)
    if mask_bool.ndim == 3:
        mask_bool = mask_bool.squeeze()

    # Known mask grows as we fill
    known_mask = ~mask_bool.copy()

    crack_coords = list(zip(*np.where(mask_bool)))
    if not crack_coords:
        return original_img.astype(np.uint8)

    print(f"Median + Weighted Blend inpainting on {len(crack_coords)} pixels (neighbor_weight={neighbor_weight})...")

    neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

    filled = 0
    for py, px in crack_coords:
        half_size = 0
        candidates_values = []

        while half_size <= (max_size // 2):
            y1 = max(0, py - half_size)
            y2 = min(h, py + half_size + 1)
            x1 = max(0, px - half_size)
            x2 = min(w, px + half_size + 1)

            square_area = (y2 - y1) * (x2 - x1)
            if square_area <= 1:
                half_size += step
                continue

            # Collect values from currently known pixels in square
            current_candidates = []
            for ky in range(y1, y2):
                for kx in range(x1, x2):
                    if known_mask[ky, kx]:
                        current_candidates.append(img[ky, kx])

            if len(current_candidates) >= ratio * square_area:
                candidates_values = np.array(current_candidates)
                break

            half_size += step

        # Fallback: use median of all known pixels if no local candidates
        if len(candidates_values) == 0:
            all_known = img[known_mask]
            if len(all_known) > 0:
                median_value = np.median(all_known, axis=0)
            else:
                median_value = np.zeros(channels)
        else:
            median_value = np.median(candidates_values, axis=0)

        # Get 8-neighbor values (already known, including previously filled)
        neighbor_values = []
        valid_neighbor_coords = []
        for dy, dx in neighbors:
            ny, nx = py + dy, px + dx
            if 0 <= ny < h and 0 <= nx < w and known_mask[ny, nx]:
                neighbor_values.append(img[ny, nx])
                valid_neighbor_coords.append((ny, nx))

        num_neighbors = len(neighbor_values)

        # Weighted blend
        final_value = median_value * (1.0 - neighbor_weight)

        if num_neighbors > 0:
            neighbor_contrib = np.mean(neighbor_values, axis=0)  # or could use median too
            final_value += neighbor_contrib * neighbor_weight
        # else: fully rely on median from square

        # Assign
        img[py, px] = final_value

        # Update known mask
        known_mask[py, px] = True

        filled += 1
        if filled % 1000 == 0:
            print(f"  Progress: {filled}/{len(crack_coords)}")

    # === Final Gaussian smoothing only on original crack pixels ===
    if gaussian_ksize > 1:
        print(f"Applying final targeted Gaussian (ksize={gaussian_ksize}, sigma={gaussian_sigma})...")
        blurred = cv2.GaussianBlur(img, (gaussian_ksize, gaussian_ksize), gaussian_sigma)

        if channels == 1:
            crack_mask_3d = mask_bool
        else:
            crack_mask_3d = np.repeat(mask_bool[..., np.newaxis], channels, axis=2)

        img[crack_mask_3d] = blurred[crack_mask_3d]

    result = np.clip(img, 0, 255).astype(np.uint8)
    if was_grayscale:
        result = result.squeeze()

    return result

# ==================== Test Your Ultimate Method ====================
def test_ultimate_inpainting(image_path, output_dir="ultimate_results"):
    os.makedirs(output_dir, exist_ok=True)

    original, final_mask = crack_detection(image_path, output_dir)

    mask_dilated = final_mask.copy()

    print("Running your ULTIMATE method: Median candidates + Neighbor blending...")
    inpainted = inpaint_median_weighted_blend(original, mask_dilated,
                                              step=6,
                                              ratio=0.05,
                                              max_size=201,
                                              neighbor_weight=0.8,     # 60% from neighbors, 40% from median
                                              gaussian_ksize=25,
                                              gaussian_sigma=0.1)

    cv2.imwrite(os.path.join(output_dir, "ultimate_crack_removed.jpg"), inpainted)

    plt.figure(figsize=(15, 8))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Mask")
    plt.imshow(mask_dilated, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Ultimate Result:\nMedian Texture + Neighbor Blend")
    plt.imshow(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    # plt.show()

    return inpainted


if __name__ == "__main__":
    image_path = "D:/University/Computer Vision/BTL/example/crack.jpg"
    test_ultimate_inpainting(image_path)