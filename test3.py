import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import random

# ==================== All Inpainting Approaches ====================


# 1. Diffusion-Based (OpenCV Telea / Navier-Stokes)
def inpaint_diffusion(original_img, mask, radius=5, method="telea"):
    kernel = np.ones((5, 5), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=2)

    flags = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(original_img, mask_dilated, inpaintRadius=radius, flags=flags)


# 2. Exemplar-Based (Patch-Based) From Scratch - Best for Pavement Texture
def inpaint_exemplar(
    original_img, mask, patch_size=13, search_samples=800, max_iters=100000
):
    img = original_img.copy().astype(np.float32)
    h, w = img.shape[:2]

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).astype(np.float32)
        was_grayscale = True
    else:
        was_grayscale = False

    half = patch_size // 2
    img_padded = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_REFLECT)

    mask_bool = mask > 127
    if mask_bool.ndim == 3:
        mask_bool = mask_bool[:, :, 0]

    mask_padded = cv2.copyMakeBorder(
        mask_bool.astype(np.uint8), half, half, half, half, cv2.BORDER_CONSTANT, value=0
    )
    mask_padded = mask_padded > 127

    to_fill = list(zip(*np.where(mask_bool)))

    source_centers = [
        (y, x)
        for y in range(half, h + half)
        for x in range(half, w + half)
        if not mask_padded[y, x]
    ]

    filled = 0
    total = len(to_fill)
    print(f"Exemplar inpainting: {total} pixels to fill...")

    while to_fill and filled < max_iters:
        candidates = []
        for py, px in to_fill:
            known_count = sum(
                1
                for dy, dx in [
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                    (0, -1),
                    (0, 1),
                    (1, -1),
                    (1, 0),
                    (1, 1),
                ]
                if 0 <= py + dy < h
                and 0 <= px + dx < w
                and not mask_bool[py + dy, px + dx]
            )
            candidates.append((known_count, py, px))

        candidates.sort(reverse=True, key=lambda x: x[0])
        _, py, px = candidates[0]

        py_pad = py + half
        px_pad = px + half

        ty = slice(py_pad - half, py_pad + half + 1)
        tx = slice(px_pad - half, px_pad + half + 1)

        target_patch = img_padded[ty, tx]
        valid_mask_2d = ~mask_padded[ty, tx]
        fill_region_2d = mask_padded[ty, tx]

        valid_mask_3d = valid_mask_2d[:, :, np.newaxis]

        best_source = random.choice(source_centers)
        if np.any(valid_mask_2d):
            samples = random.sample(
                source_centers, min(search_samples, len(source_centers))
            )
            best_dist = np.inf
            for sy_pad, sx_pad in samples:
                source_patch = img_padded[
                    sy_pad - half : sy_pad + half + 1, sx_pad - half : sx_pad + half + 1
                ]
                diff = source_patch - target_patch
                dist = np.sum((diff**2) * valid_mask_3d) / (
                    np.sum(valid_mask_2d) + 1e-8
                )
                if dist < best_dist:
                    best_dist = dist
                    best_source = (sy_pad, sx_pad)

        sy_pad, sx_pad = best_source
        source_patch = img_padded[
            sy_pad - half : sy_pad + half + 1, sx_pad - half : sx_pad + half + 1
        ]

        img_padded[ty, tx][fill_region_2d] = source_patch[fill_region_2d]

        mask_bool[py, px] = False
        to_fill.remove((py, px))

        filled += 1
        if filled % 2000 == 0 or filled == total:
            print(f"  Progress: {filled}/{total}")

    result = np.clip(img_padded[half : h + half, half : w + half], 0, 255).astype(
        np.uint8
    )
    if was_grayscale:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    return result


# 3. Morphological Filling
def inpaint_morphological(original_img, mask, close_ksize=21):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    closed = cv2.morphologyEx(original_img, cv2.MORPH_CLOSE, kernel, iterations=3)

    result = original_img.copy()
    result[mask > 127] = closed[mask > 127]
    return result


# 4. Iterative Median + Gaussian Filling
def inpaint_median_gaussian(original_img, mask, iterations=100):
    result = original_img.copy().astype(np.float32)
    mask_bool = mask > 127

    for i in range(iterations):
        median = cv2.medianBlur(result.astype(np.uint8), 9)
        blurred = cv2.GaussianBlur(median, (5, 5), 0)
        result[mask_bool] = blurred[mask_bool]

    return np.clip(result, 0, 255).astype(np.uint8)


# ==================== Crack Detection (Your Pipeline) ====================
# (Paste your full crack_detection function here - unchanged)

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


# ==================== Test All Approaches ====================
def test_all_inpainting_methods(image_path, output_dir="results_all"):
    os.makedirs(output_dir, exist_ok=True)

    original_color, final_mask = crack_detection(
        image_path, output_dir
    )  # Your function

    # # Dilate mask for better coverage
    # kernel = np.ones((5, 5), np.uint8)
    # mask_dilated = cv2.dilate(final_mask, kernel, iterations=2)
    # cv2.imwrite(os.path.join(output_dir, "mask_dilated.png"), mask_dilated)

    mask_dilated = final_mask  # Use the original mask from crack_detectionj

    print("Testing all inpainting approaches...")

    results = {}

    # 1. Diffusion (Telea)
    results["Diffusion (Telea)"] = inpaint_diffusion(
        original_color, mask_dilated, method="telea"
    )

    # 2. Diffusion (Navier-Stokes)
    results["Diffusion (NS)"] = inpaint_diffusion(
        original_color, mask_dilated, method="ns"
    )

    # 3. Exemplar-Based (Best for texture)
    results["Exemplar-Based"] = inpaint_exemplar(
        original_color, mask_dilated, patch_size=13, search_samples=800
    )

    # 4. Morphological
    results["Morphological"] = inpaint_morphological(original_color, mask_dilated)

    # 5. Median + Gaussian
    results["Median + Gaussian"] = inpaint_median_gaussian(
        original_color, mask_dilated, iterations=150
    )

    # Save all
    for name, img in results.items():
        path = os.path.join(output_dir, f"{name.replace(' ', '_').lower()}.jpg")
        cv2.imwrite(path, img)
        print(f"Saved: {path}")

    # Visualization
    plt.figure(figsize=(15, 12))
    plt.subplot(3, 2, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(original_color, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.subplot(3, 2, 2)
    plt.title("Crack Mask")
    plt.imshow(mask_dilated, cmap="gray")
    plt.axis("off")
    for i, (name, img) in enumerate(results.items(), 3):
        plt.subplot(3, 2, i)
        plt.title(name)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = "D:/University/Computer Vision/BTL/example/065.jpg"  # Your path
    image_path = "real_life_image/jpg/2025_12_27_17_45_IMG_6463.jpg"
    test_all_inpainting_methods(image_path)
