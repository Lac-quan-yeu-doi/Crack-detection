# ===================================================================
# ULTIMATE CLASSICAL CRACK DETECTION SUITE (10 METHODS)
# All methods: NO TRAINING, NO DEEP LEARNING, ONLY IMAGE PROCESSING
# Added: Sobel, Canny, LoG, Gabor, Morphological (Top/Bottom Hat)
# ===================================================================
import os
import cv2
import numpy as np
from skimage.filters import frangi, threshold_otsu, threshold_sauvola, gabor
from skimage.morphology import (
    skeletonize,
    disk,
    reconstruction,
    remove_small_objects,
    closing,
)
from scipy.ndimage import gaussian_filter, convolve, laplace
from skimage.graph import route_through_array
import pywt
from skimage.segmentation import felzenszwalb
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import matplotlib.pyplot as plt


# ----------------------- YOUR ORIGINAL 5 APPROACHES (UNCHANGED) -----------------------
# (approach1_frangi, approach2_hessian_minpath, approach3_bottomhat_wavelet_graph,
#  approach4_tophat_region_growing, approach5_phase_congruency)
# ... [paste your original 5 functions here exactly as they are] ...
# I'll keep them collapsed for space, but they are 100% unchanged


def approach1_frangi(img):
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    img_enh = clahe.apply(img)
    sigmas = np.arange(2, 12, 1)
    frangi_map = frangi(
        img_enh, sigmas=sigmas, alpha=0.7, beta=0.3, gamma=15, black_ridges=True
    )
    frangi_map = np.nan_to_num(frangi_map)
    frangi_map = (frangi_map - frangi_map.min()) / (frangi_map.ptp() + 1e-8)
    thresh = threshold_sauvola(frangi_map, window_size=35, k=0.15)
    binary = (frangi_map > thresh).astype(np.uint8) * 255
    binary = remove_small_objects(binary > 0, min_size=100, connectivity=2)
    binary = binary.astype(np.uint8) * 255
    skeleton = skeletonize(binary > 0)
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result[skeleton] = (0, 0, 255)
    return result, skeleton.astype(np.uint8) * 255


def approach2_hessian_minpath(img):
    blur = gaussian_filter(img.astype(float), sigma=1.0)
    Ix = convolve(blur, np.array([[-1, 0, 1]]))
    Iy = convolve(blur, np.array([[-1], [0], [1]]))
    Ixx = convolve(blur, np.array([[1, -2, 1]]))
    Iyy = convolve(blur, np.array([[1], [-2], [1]]))
    Ixy = convolve(Ix, np.array([[1], [-2], [1]])) / 4
    discriminant = Ixx * Iyy - Ixy * Ixy
    trace = Ixx + Iyy
    lambda2 = 0.5 * (trace - np.sqrt(np.abs(trace**2 - 4 * discriminant + 1e-8)))
    response = -lambda2
    response = np.nan_to_num(response)
    response = (response - response.min()) / (response.ptp() + 1e-8)
    thresh = np.percentile(response, 92)
    binary = (response > thresh).astype(np.uint8) * 255
    cost = 1.0 - response
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    linked = binary.copy()
    for cnt in contours:
        if len(cnt) > 20:
            start = tuple(cnt[0][0])
            end = tuple(cnt[-1][0])
            try:
                path, _ = route_through_array(cost, start, end, fully_connected=True)
                for y, x in path:
                    linked[y, x] = 255
            except:
                pass
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result[linked > 0] = (0, 255, 0)
    return result, linked


def approach3_bottomhat_wavelet_graph(img):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_cl = clahe.apply(img)
    se = disk(5)
    enhanced = cv2.morphologyEx(img_cl, cv2.MORPH_BLACKHAT, se)
    coeffs = pywt.wavedec2(enhanced, "db4", level=2)
    coeffs = list(coeffs)
    coeffs[1:] = [
        tuple(pywt.threshold(c, np.std(c) * 1.2, mode="hard") for c in level)
        for level in coeffs[1:]
    ]
    denoised = pywt.waverec2(coeffs, "db4")
    denoised = np.clip(denoised, 0, 255).astype(np.uint8)
    if denoised.shape != img.shape:
        min_h, min_w = min(denoised.shape[0], img.shape[0]), min(
            denoised.shape[1], img.shape[1]
        )
        denoised = denoised[:min_h, :min_w]
        img_local = img[:min_h, :min_w]
    else:
        img_local = img
    rgb = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    segments = felzenszwalb(rgb, scale=25, sigma=0.8, min_size=60)
    mask = np.zeros_like(denoised, dtype=np.uint8)
    avg = np.mean(denoised)
    for seg_id in np.unique(segments):
        region = segments == seg_id
        if np.mean(denoised[region]) < avg * 0.95:
            mask[region] = 255
    mask = remove_small_objects(mask > 0, min_size=40).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    mask = cv2.dilate(
        mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
    )
    result = cv2.cvtColor(img_local, cv2.COLOR_GRAY2BGR)
    result[mask > 0] = (255, 0, 0)
    return result, mask


def approach4_tophat_region_growing(img):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_cl = clahe.apply(img)
    se = disk(7)
    tophat = cv2.morphologyEx(img_cl, cv2.MORPH_TOPHAT, se)
    enhanced = 255 - tophat
    thresh_seed = np.mean(enhanced) - 1.0 * np.std(enhanced)
    seeds = enhanced < thresh_seed
    seeds = seeds.astype(np.uint8)
    marker = (enhanced.astype(float) * 0.90).astype(enhanced.dtype)
    marker[seeds == 0] = 0
    mask_rec = reconstruction(marker, enhanced, method="dilation")
    binary = (enhanced > (mask_rec * 0.85)).astype(np.uint8) * 255
    binary = remove_small_objects(binary > 0, min_size=50).astype(np.uint8) * 255
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_DILATE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    result = cv2.cvtColor(img_cl, cv2.COLOR_GRAY2BGR)
    result[binary > 0] = (0, 255, 255)
    return result, binary


def approach5_phase_congruency(img):
    img_f = img.astype(float)
    response = np.zeros_like(img_f, dtype=float)
    for sigma in [1, 2, 3]:
        H_elems = hessian_matrix(img_f, sigma=sigma, order="xy")
        eigvals = hessian_matrix_eigvals(H_elems)
        response += np.abs(eigvals[1])
    response = (response - response.min()) / (response.ptp() + 1e-8)
    response = gaussian_filter(response, sigma=0.8)
    thresh = np.percentile(response, 80)
    binary = (response > thresh).astype(np.uint8) * 255
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    binary = remove_small_objects(binary > 0, min_size=30).astype(np.uint8) * 255
    sk = skeletonize(binary > 0).astype(np.uint8) * 255
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result[sk > 0] = (255, 165, 0)
    return result, sk


# ----------------------- NEW: 5 CLASSICAL BASELINES (6–10) -----------------------


def approach6_sobel_morphology(img):
    """Sobel gradient + Otsu + Morphology cleanup"""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_cl = clahe.apply(img)
    sobelx = cv2.Sobel(img_cl, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_cl, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))
    _, binary = cv2.threshold(magnitude, 0, 255, cv2.THRESH_OTSU)
    binary = closing(binary, disk(3))
    binary = remove_small_objects(binary > 0, min_size=50).astype(np.uint8) * 255
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result[binary > 0] = (255, 192, 203)  # Pink
    return result, binary


def approach7_canny_hysteresis(img):
    """Standard Canny with adaptive thresholds + closing"""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_cl = clahe.apply(img)
    # Adaptive thresholds using median
    med = np.median(img_cl)
    low = int(max(0, 0.7 * med))
    high = int(min(255, 1.3 * med))
    edges = cv2.Canny(img_cl, low, high, L2gradient=True)
    edges = closing(edges, disk(3))
    edges = remove_small_objects(edges > 0, min_size=50).astype(np.uint8) * 255
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result[edges > 0] = (0, 165, 255)  # Orange-red
    return result, edges


def approach8_log_zero_crossing(img):
    """Laplacian of Gaussian + Zero Crossing"""
    blur = gaussian_filter(img.astype(float), sigma=2.0)
    log = laplace(blur)
    # Zero-crossing detection
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    log_manual = convolve(blur, kernel)
    zero_cross = np.zeros_like(img, dtype=np.uint8)
    h, w = img.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            patch = log_manual[y - 1 : y + 2, x - 1 : x + 2]
            if np.min(patch) < 0 and np.max(patch) > 0:
                zero_cross[y, x] = 255
    zero_cross = (
        remove_small_objects(zero_cross > 0, min_size=60).astype(np.uint8) * 255
    )
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result[zero_cross > 0] = (128, 0, 128)  # Purple
    return result, zero_cross


def approach9_gabor_bank(img):
    """Multi-orientation Gabor filter bank"""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_cl = clahe.apply(img).astype(float)
    response = np.zeros_like(img_cl)
    frequencies = [0.1, 0.2]
    thetas = np.arange(0, np.pi, np.pi / 8)  # 8 directions
    for freq in frequencies:
        for theta in thetas:
            real, imag = gabor(img_cl, frequency=freq, theta=theta)
            response = np.maximum(response, np.sqrt(real**2 + imag**2))
    response = (response - response.min()) / (response.ptp() + 1e-8)
    thresh = threshold_otsu(response)
    binary = (response > thresh).astype(np.uint8) * 255
    binary = closing(binary, disk(3))
    binary = remove_small_objects(binary > 0, min_size=60).astype(np.uint8) * 255
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result[binary > 0] = (255, 255, 0)  # Cyan
    return result, binary


def approach10_morphological_combo(img):
    """Top-Hat + Bottom-Hat + Thresholding (very robust baseline)"""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_cl = clahe.apply(img)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    tophat = cv2.morphologyEx(img_cl, cv2.MORPH_TOPHAT, se)
    bothat = cv2.morphologyEx(img_cl, cv2.MORPH_BLACKHAT, se)
    enhanced = img_cl + tophat - bothat
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_OTSU)
    binary = closing(binary, disk(5))
    binary = remove_small_objects(binary > 0, min_size=80).astype(np.uint8) * 255
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result[binary > 0] = (0, 255, 255)  # Yellow
    return result, binary


# ===================================================================
# MAIN – Run ALL 10 approaches + beautiful comparison
# ===================================================================
if __name__ == "__main__":
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    image_path = (
        "dataset/Concrete & Pavement Crack Dataset/Positive/00020.jpg"  # CHANGE THIS
    )
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image not found! Check path.")

    # Run all 10
    results = []
    masks = []
    names = [
        "Original",
        "1. Frangi",
        "2. Hessian+Path",
        "3. BottomHat+Graph",
        "4. TopHat+Region",
        "5. Phase Congruency",
        "6. Sobel+Morph",
        "7. Canny",
        "8. LoG ZeroCross",
        "9. Gabor Bank",
        "10. Morph Combo",
    ]

    # Original image
    results.append(img)
    masks.append(None)

    # Your 5 original
    r, m = approach1_frangi(img)
    results.append(r)
    masks.append(m)
    r, m = approach2_hessian_minpath(img)
    results.append(r)
    masks.append(m)
    r, m = approach3_bottomhat_wavelet_graph(img)
    results.append(r)
    masks.append(m)
    r, m = approach4_tophat_region_growing(img)
    results.append(r)
    masks.append(m)
    r, m = approach5_phase_congruency(img)
    results.append(r)
    masks.append(m)

    # New 5 baselines
    r, m = approach6_sobel_morphology(img)
    results.append(r)
    masks.append(m)
    r, m = approach7_canny_hysteresis(img)
    results.append(r)
    masks.append(m)
    r, m = approach8_log_zero_crossing(img)
    results.append(r)
    masks.append(m)
    r, m = approach9_gabor_bank(img)
    results.append(r)
    masks.append(m)
    r, m = approach10_morphological_combo(img)
    results.append(r)
    masks.append(m)

    # Save all masks
    mask_names = [
        "frangi",
        "hessian",
        "wavelet_graph",
        "tophat_region",
        "phase",
        "sobel",
        "canny",
        "log",
        "gabor",
        "morph_combo",
    ]
    for name, mask in zip(mask_names, masks[1:]):
        cv2.imwrite(f"{output_dir}/mask_{name}.png", mask)

    plt.figure(figsize=(48, 30))

    # total images = 11 results + 10 masks = 21
    cols = 7

    # ---- top 11 images ----
    for i in range(11):
        plt.subplot(3, cols, i + 1)
        if i == 0:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(cv2.cvtColor(results[i], cv2.COLOR_BGR2RGB))
        plt.title(names[i], fontsize=12)
        plt.axis("off")

    # ---- bottom 10 masks ----
    for i in range(10):
        plt.subplot(3, cols, 11 + i + 1)
        plt.imshow(masks[i + 1], cmap="gray")
        plt.title(f"Mask {i+1}", fontsize=10)
        plt.axis("off")


    plt.suptitle("10 Classical Crack Detection Methods Comparison", fontsize=20, y=0.98)
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/ULTIMATE_10_METHODS_COMPARISON.jpg", dpi=400, bbox_inches="tight"
    )
    plt.show()

    print("ALL 10 METHODS DONE! Check output folder.")
