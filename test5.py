# random choose pixels from growing square + weighted blend + final Gaussian

import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import random


def crack_detection(image_input, output_dir="result"):
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(image_input, str):
        original_color = cv2.imread(image_input)
        gray = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Could not read image from path: {image_input}")
    # If input is image array
    elif isinstance(image_input, np.ndarray):
        original_color = image_input.copy()
        if image_input.ndim == 3:  # Color image (BGR)
            gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        elif image_input.ndim == 2:  # Already grayscale
            gray = image_input
        else:
            raise ValueError(
                f"Invalid image array shape: {image_input.shape}. Expected 2D or 3D."
            )
    else:
        raise TypeError(
            f"Unsupported input type: {type(image_input)}. Must be str (path) or np.ndarray (image)."
        )

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


def inpaint_random_weighted_blend(
    original_img,
    mask,
    step=5,
    ratio=0.05,
    max_size=151,
    main_weight=0.4,
    gaussian_ksize=9,  # Final Gaussian on filled regions
    gaussian_sigma=1.2,
    random_seed=None,
):
    """
    - Randomly select a texture pixel from growing square
    - Blend it with 8-neighbors (including previously filled pixels)
    - main_weight for random pixel, rest shared among neighbors
    - Final Gaussian smooth on crack areas only
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    img = original_img.copy().astype(np.float32)
    h, w = img.shape[:2]

    if img.ndim == 2:
        img = img[..., np.newaxis]
        was_grayscale = True
        channels = 1
    else:
        was_grayscale = False
        channels = img.shape[2]

    # Original mask — we fill these pixels
    mask_bool = mask > 127
    if mask_bool.ndim == 3:
        mask_bool = mask_bool.squeeze()

    # Known pixels initially = non-crack
    known_mask = ~mask_bool  # Will grow as we fill

    crack_coords = list(zip(*np.where(mask_bool)))
    if not crack_coords:
        return original_img.astype(np.uint8)

    print(
        f"Inpainting on {len(crack_coords)} crack pixels (main_weight={main_weight})..."
    )

    # 8-connectivity offsets
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    filled = 0
    for py, px in crack_coords:
        # Grow square to find candidate known pixels
        half_size = 0
        candidates = []

        while half_size <= (max_size // 2) and not candidates:
            y1 = max(0, py - half_size)
            y2 = min(h, py + half_size + 1)
            x1 = max(0, px - half_size)
            x2 = min(w, px + half_size + 1)

            square_area = (y2 - y1) * (x2 - x1)
            if square_area <= 1:
                half_size += step
                continue

            # Collect currently known pixels in square
            for ky in range(y1, y2):
                for kx in range(x1, x2):
                    if known_mask[ky, kx]:
                        candidates.append((ky, kx))

            if len(candidates) < ratio * square_area:
                candidates = []  # Not enough yet

            half_size += step

        # If still no candidates, use any known pixel in image
        if not candidates:
            known_flat = list(zip(*np.where(known_mask)))
            if known_flat:
                candidates = [random.choice(known_flat)]

        if not candidates:
            filled += 1
            continue

        # Randomly select one main texture pixel
        main_ky, main_kx = random.choice(candidates)
        main_value = img[main_ky, main_kx]

        # Collect 8-neighbors that are already known
        neighbor_values = []
        for dy, dx in neighbors:
            ny, nx = py + dy, px + dx
            if 0 <= ny < h and 0 <= nx < w and known_mask[ny, nx]:
                neighbor_values.append(img[ny, nx])

        num_neighbors = len(neighbor_values)
        blended = main_value * main_weight

        if num_neighbors > 0:
            neighbor_weight = (1.0 - main_weight) / num_neighbors
            for val in neighbor_values:
                blended += val * neighbor_weight
        else:
            # If no neighbors, just use main with full weight
            blended = main_value

        # Assign blended value
        img[py, px] = blended

        # Now this pixel is known for future fills
        known_mask[py, px] = True

        filled += 1
        if filled % 1000 == 0:
            print(f"  Progress: {filled}/{len(crack_coords)}")

    # === Final: Gaussian smooth only on original crack regions ===
    if gaussian_ksize > 1:
        print(
            f"Applying final targeted Gaussian (ksize={gaussian_ksize}, sigma={gaussian_sigma})..."
        )
        blurred = cv2.GaussianBlur(
            img, (gaussian_ksize, gaussian_ksize), gaussian_sigma
        )

        # Mask for original crack pixels
        if channels == 1:
            crack_mask_3d = mask_bool
        else:
            crack_mask_3d = np.repeat(mask_bool[..., np.newaxis], channels, axis=2)

        img[crack_mask_3d] = blurred[crack_mask_3d]

    result = np.clip(img, 0, 255).astype(np.uint8)
    if was_grayscale:
        result = result.squeeze()

    return result


# ==================== Test Your Best Method ====================
def test_smart_inpainting(image_path, output_dir="smart_results"):
    os.makedirs(output_dir, exist_ok=True)
    import utils

    down = utils.downsample(cv2.imread(image_path), 0.2)
    # original, final_mask = crack_detection(down, output_dir)
    original, final_mask = crack_detection(image_path, output_dir)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_dilated = cv2.dilate(final_mask, kernel, iterations=2)
    # mask_dilated = final_mask.copy()

    print("Running your SMART weighted blend inpainting...")
    inpainted = inpaint_random_weighted_blend(
        original,
        mask_dilated,
        step=6,
        ratio=0.05,
        max_size=101,
        main_weight=0.9,  # 40% from random texture
        gaussian_ksize=15,
        gaussian_sigma=0.5,
        random_seed=50,
    )

    cv2.imwrite(os.path.join(output_dir, "smart_crack_removed.jpg"), inpainted)

    plt.figure(figsize=(15, 8))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Mask")
    plt.imshow(mask_dilated, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Your Smart Method\nWeighted Blend + Gaussian")
    plt.imshow(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    return inpainted


if __name__ == "__main__":
    image_path = "D:/University/Computer Vision/BTL/example/065.jpg"
    # image_path = "real_life_image/jpg/2025_12_27_17_45_IMG_6463.jpg"
    test_smart_inpainting(image_path)
