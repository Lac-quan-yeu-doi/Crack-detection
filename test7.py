import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def overlay_mask_on_image(image, mask, color=(255, 0, 0), alpha=0.5):
    """
    Overlay binary mask on image.
    - image: BGR uint8 image
    - mask: binary or grayscale mask (0 or 255)
    - color: overlay color in BGR
    - alpha: transparency (0–1)
    """
    overlay = image.copy()
    mask_bool = mask > 0

    # Apply color only where mask is True
    overlay[mask_bool] = (
        (1 - alpha) * overlay[mask_bool] +
        alpha * np.array(color, dtype=np.uint8)
    )

    return overlay.astype(np.uint8)

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
            raise ValueError(f"Invalid image array shape: {image_input.shape}. Expected 2D or 3D.")
    else:
        raise TypeError(f"Unsupported input type: {type(image_input)}. Must be str (path) or np.ndarray (image).")
    
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

def inpaint_bitplane_reflection(original_img, mask,
                                window_size=21,         # Odd: window side
                                stride=5,               # Shift step
                                priority='horizontal',  # 'horizontal' or 'vertical' first
                                gaussian_ksize=11,      # Final Gaussian kernel (odd)
                                gaussian_sigma=1.0):    # Sigma (0 = auto)
    """
    Your bit-plane reflection mirroring inpainting.
    - Per bit-plane per channel
    - Slide window; for each crack in window, reflect over priority axis
    - If source is crack, try other direction
    - Final targeted Gaussian on crack areas
    """
    img = original_img.copy().astype(np.uint8)
    h, w = img.shape[:2]

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).astype(np.uint8)
        was_grayscale = True
        channels = 3
    else:
        was_grayscale = False
        channels = img.shape[2]

    mask_bool = (mask > 127)
    if mask_bool.ndim == 3:
        mask_bool = mask_bool.squeeze()

    # Original mask for final blur
    original_mask_bool = mask_bool.copy()

    half_win = window_size // 2

    # Directions
    directions = ['horizontal', 'vertical'] if priority == 'horizontal' else ['vertical', 'horizontal']

    # Process each channel
    result = np.zeros_like(img)
    for c in range(channels):
        channel = img[:, :, c]

        # 8 bit planes
        bit_planes = [(channel & (1 << b)) >> b for b in range(8)]

        processed_planes = []
        for bp in bit_planes:
            bp_processed = bp.copy()
            mask_processed = mask_bool.copy()  # Reset per bit plane? No, shared fills, but since bit independent, but to propagate, use per bp

            for y in range(half_win, h - half_win, stride):
                for x in range(half_win, w - half_win, stride):
                    y1 = y - half_win
                    y2 = y + half_win + 1
                    x1 = x - half_win
                    x2 = x + half_win + 1

                    mask_window = mask_processed[y1:y2, x1:x2]

                    if not np.any(mask_window):
                        continue

                    for ry in range(window_size):
                        for rx in range(window_size):
                            if mask_window[ry, rx]:
                                py = y1 + ry
                                px = x1 + rx

                                filled = False
                                for dir in directions:
                                    if dir == 'horizontal':  # Flip left-right over vertical center
                                        refl_rx = 2 * half_win - rx
                                        refl_ry = ry
                                    else:  # Flip up-down over horizontal center
                                        refl_rx = rx
                                        refl_ry = 2 * half_win - ry

                                    # Source absolute
                                    source_y = y1 + refl_ry
                                    source_x = x1 + refl_rx

                                    if 0 <= source_y < h and 0 <= source_x < w:
                                        if not mask_processed[source_y, source_x]:
                                            bp_processed[py, px] = bp[source_y, source_x]
                                            mask_processed[py, px] = False
                                            filled = True
                                            break

            processed_planes.append(bp_processed)

        # Reconstruct channel
        reconstructed = np.zeros((h, w), dtype=np.uint8)
        for b in range(8):
            reconstructed |= (processed_planes[b] << b)

        result[:, :, c] = reconstructed

    # === Final Gaussian smoothing on original crack regions ===
    if gaussian_ksize > 1:
        print(f"Applying targeted Gaussian blur (ksize={gaussian_ksize}, sigma={gaussian_sigma}) to crack regions...")
        blurred = cv2.GaussianBlur(result, (gaussian_ksize, gaussian_ksize), gaussian_sigma)

        if channels == 1:
            crack_mask_3d = original_mask_bool
        else:
            crack_mask_3d = np.repeat(original_mask_bool[..., np.newaxis], channels, axis=2)

        result[crack_mask_3d] = blurred[crack_mask_3d]

    if was_grayscale:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    return result

# ==================== Test Function ====================
def test_bitplane_reflection_inpainting(image_path, output_dir="bitplane_results"):
    os.makedirs(output_dir, exist_ok=True)

    import utils
    img = cv2.imread(image_path)
    down = utils.downsample(img, 0.4)
    original, final_mask = crack_detection(down, output_dir)   
    
    # original, final_mask = crack_detection(image_path, output_dir)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask_dilated = cv2.dilate(final_mask, kernel, iterations=2)

    print("Running your BIT-PLANE REFLECTION INPAINTING...")
    inpainted = inpaint_bitplane_reflection(original, mask_dilated,
                                            window_size=45,
                                            stride=5,
                                            priority='vertical',
                                            gaussian_ksize=3,
                                            gaussian_sigma=0.2)

    cv2.imwrite(os.path.join(output_dir, "bitplane_result.jpg"), inpainted)

    overlay = overlay_mask_on_image(original, mask_dilated,
                                color=(0, 0, 255),  # red in BGR
                                alpha=0.5)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 4, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("Crack Mask")
    plt.imshow(mask_dilated, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("Mask Overlay")
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("Bit-Plane Inpainting Result")
    plt.imshow(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return inpainted


if __name__ == "__main__":
    image_path = "D:/University/Computer Vision/BTL/example/065.jpg"
    image_path = "real_life_image/jpg/2025_12_27_17_45_IMG_6463.jpg"
    image_path = "real_life_image/jpg/2025_12_27_17_49_IMG_6476.jpg"
    # image_path = "D:/University/Computer Vision/BTL/example/crack2.jpg"
    test_bitplane_reflection_inpainting(image_path)