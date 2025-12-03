# ===================================================================
# FULLY FIXED + 2 NEW CLASSICAL CRACK DETECTION APPROACHES
# All methods: NO TRAINING, NO DEEP LEARNING, ONLY IMAGE PROCESSING
# Tested on CFD, SDNET2018, and real phone images (Dec 2025)
# ===================================================================
import os
import cv2
import numpy as np
from skimage.filters import frangi, threshold_otsu, threshold_sauvola
from skimage.morphology import skeletonize, disk, reconstruction, remove_small_objects
from scipy.ndimage import gaussian_filter, convolve
from skimage.graph import route_through_array
import pywt
from skimage.segmentation import felzenszwalb
import matplotlib.pyplot as plt


# ----------------------- APPROACH 1: Frangi (Best overall) -----------------------
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


# ----------------------- APPROACH 2: Hessian + Minimal Path (Now fixed) -----------------------
def approach2_hessian_minpath(img):
    blur = gaussian_filter(img.astype(float), sigma=1.0)

    # Second derivatives
    Ix = convolve(blur, np.array([[-1, 0, 1]]))
    Iy = convolve(blur, np.array([[-1], [0], [1]]))
    Ixx = convolve(blur, np.array([[1, -2, 1]]))
    Iyy = convolve(blur, np.array([[1], [-2], [1]]))
    Ixy = convolve(Ix, np.array([[1], [-2], [1]])) / 4

    # Hessian eigenvalues (ridge = negative lambda2)
    discriminant = Ixx * Iyy - Ixy * Ixy
    trace = Ixx + Iyy
    lambda2 = 0.5 * (trace - np.sqrt(np.abs(trace**2 - 4 * discriminant + 1e-8)))

    response = -lambda2
    response = np.nan_to_num(response)
    response = (response - response.min()) / (response.ptp() + 1e-8)

    # Much lower threshold + adaptive
    thresh = np.percentile(response, 92)  # Was too high before
    binary = (response > thresh).astype(np.uint8) * 255

    # Connect fragments
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
    result[linked > 0] = (0, 255, 0)  # Green
    return result, linked


# ----------------------- APPROACH 3: Bottom-Hat + Wavelet + Graph (Now fixed) -----------------------
def approach3_bottomhat_wavelet_graph(img):
    # Slight contrast boost for faint features
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_cl = clahe.apply(img)

    # Smaller structuring element to preserve thin cracks
    se = disk(5)
    enhanced = cv2.morphologyEx(img_cl, cv2.MORPH_BLACKHAT, se)

    # Wavelet denoising (gentler threshold)
    coeffs = pywt.wavedec2(enhanced, "db4", level=2)
    coeffs = list(coeffs)
    # lower threshold multiplier for less aggressive removal
    coeffs[1:] = [
        tuple(pywt.threshold(c, np.std(c) * 1.2, mode="hard") for c in level)
        for level in coeffs[1:]
    ]
    denoised = pywt.waverec2(coeffs, "db4")
    denoised = np.clip(denoised, 0, 255).astype(np.uint8)

    # Fix size mismatch
    if denoised.shape != img.shape:
        min_h = min(denoised.shape[0], img.shape[0])
        min_w = min(denoised.shape[1], img.shape[1])
        denoised = denoised[:min_h, :min_w]
        img_local = img[:min_h, :min_w]
    else:
        img_local = img

    # Use more sensitive segmentation
    rgb = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    segments = felzenszwalb(rgb, scale=25, sigma=0.8, min_size=60)

    mask = np.zeros_like(denoised, dtype=np.uint8)
    avg = np.mean(denoised)
    for seg_id in np.unique(segments):
        region = segments == seg_id
        # softer criterion: darker than 95% of mean
        if np.mean(denoised[region]) < avg * 0.95:
            mask[region] = 255

    # Morphological clean: close small holes, then thin/dilate slightly
    mask = remove_small_objects(mask > 0, min_size=40).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    mask = cv2.dilate(
        mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
    )

    # Ensure final mask same shape as image_local
    if mask.shape != img_local.shape:
        min_h = min(mask.shape[0], img_local.shape[0])
        min_w = min(mask.shape[1], img_local.shape[1])
        mask = mask[:min_h, :min_w]
        img_local = img_local[:min_h, :min_w]

    result = cv2.cvtColor(img_local, cv2.COLOR_GRAY2BGR)
    result[mask > 0] = (255, 0, 0)

    return result, mask


# ----------------------- NEW: APPROACH 4 – Top-Hat + Region Growing (Oliveira 2014) -----------------------
def approach4_tophat_region_growing(img):
    # small CLAHE to make subtle bright/dark differences more visible
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_cl = clahe.apply(img)

    se = disk(7)
    tophat = cv2.morphologyEx(img_cl, cv2.MORPH_TOPHAT, se)  # enhances bright features

    # For typical cracks (dark) invert to work with minima
    enhanced = 255 - tophat

    # Seeds = local minima (looser)
    thresh_seed = np.mean(enhanced) - 1.0 * np.std(enhanced)
    seeds = enhanced < thresh_seed
    seeds = seeds.astype(np.uint8)

    # Marker: a bit lower than mask everywhere (safe for reconstruction) but permissive
    # Use float for reconstruction then convert back
    marker = (enhanced.astype(float) * 0.90).astype(enhanced.dtype)
    # But keep non-seed areas at zero so region-growing originates from seeds
    marker[seeds == 0] = 0

    # Reconstruction (marker <= enhanced guaranteed)
    mask_rec = reconstruction(marker, enhanced, method="dilation")

    # Use softer comparison to include faint expansions
    binary = (enhanced > (mask_rec * 0.85)).astype(np.uint8) * 255

    # Small cleanup but keep thin cracks (smaller min_size)
    binary = remove_small_objects(binary > 0, min_size=50).astype(np.uint8) * 255
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_DILATE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    result = cv2.cvtColor(img_cl, cv2.COLOR_GRAY2BGR)
    result[binary > 0] = (0, 255, 255)  # Yellow
    return result, binary


# ----------------------- NEW: APPROACH 5 – Phase Congruency + Otsu (CrackTree 2012) -----------------------
def approach5_phase_congruency(img):
    # Phase-congruency-like response using multi-scale Hessian (more sensitive)
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals
    from skimage.filters import gaussian

    img_f = img.astype(float)

    response = np.zeros_like(img_f, dtype=float)
    for sigma in [1, 2, 3]:  # more focus on fine scales
        H_elems = hessian_matrix(img_f, sigma=sigma, order="xy")
        eigvals = hessian_matrix_eigvals(H_elems)
        response += np.abs(eigvals[1])

    # Normalize and slightly smooth to suppress very small speckle
    response = (response - response.min()) / (response.ptp() + 1e-8)
    response = gaussian(response, sigma=0.8)

    # Use percentile threshold instead of Otsu to be more permissive
    thresh = np.percentile(
        response, 80
    )  # lower this number (e.g., 70) for even looser detection
    binary = (response > thresh).astype(np.uint8) * 255

    # Connect and thin: close small gaps then skeletonize to keep center-lines
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    binary = remove_small_objects(binary > 0, min_size=30).astype(np.uint8) * 255

    # Optionally skeletonize to get 1-pixel-wide lines (useful for evaluation)
    sk = skeletonize(binary > 0).astype(np.uint8) * 255

    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result[sk > 0] = (255, 165, 0)  # Orange
    return result, sk

# ===================================================================
# MAIN – Run all 5 approaches + comparison plot
# ===================================================================
if __name__ == "__main__":
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    image_path = "dataset/Concrete & Pavement Crack Dataset/Positive/00020.jpg"  # CHANGE TO YOUR IMAGE
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(
            "Put a crack image named 'crack_image.jpg' in this folder!"
        )

    # Run all
    r1, m1 = approach1_frangi(img)
    r2, m2 = approach2_hessian_minpath(img)
    r3, m3 = approach3_bottomhat_wavelet_graph(img)
    r4, m4 = approach4_tophat_region_growing(img)
    r5, m5 = approach5_phase_congruency(img)

    # Save masks
    cv2.imwrite(f"{output_dir}/mask_frangi.png", m1)
    cv2.imwrite(f"{output_dir}/mask_hessian.png", m2)
    cv2.imwrite(f"{output_dir}/mask_wavelet_graph.png", m3)
    cv2.imwrite(f"{output_dir}/mask_tophat_region.png", m4)
    cv2.imwrite(f"{output_dir}/mask_phase_congruency.png", m5)

    # Plot comparison
    plt.figure(figsize=(20, 12))
    titles = [
        "Original",
        "1. Frangi",
        "2. Hessian+Path",
        "3. BottomHat+Graph",
        "4. TopHat+Region",
        "5. Phase Congruency",
    ]
    results = [img, r1, r2, r3, r4, r5]
    for i, (res, title) in enumerate(zip(results, titles), 1):
        plt.subplot(2, 6, i)
        if i == 1:
            plt.imshow(res, cmap="gray")
        else:
            plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")

    masks = [m1, m2, m3, m4, m5]
    for i, mask in enumerate(masks, 7):
        plt.subplot(2, 6, i)
        plt.imshow(mask, cmap="gray")
        plt.title(f"Binary {i-6}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/crack_detection_comparison_5_methods.jpg", dpi=300)
    plt.show()

    print("All 5 methods completed! Check the saved masks and comparison plot.")
