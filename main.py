import os
import cv2
import numpy as np
from skimage.morphology import closing, skeletonize, disk, remove_small_objects
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def approach6_sobel_morphology_v1(
    img,
    clahe_clip=3.0,           # Try: 1.0, 3.0, 5.0, 10.0
    clahe_tile=(8,8),         # Try: (4,4), (8,8), (16,16)
    sobel_ksize=3,            # Try: 3, 5, 7
    closing_radius=3,         # Try: 1, 3, 5, 7
    min_object_size=100,       # Try: 20, 50, 100, 200
    thresh_method="otsu"      # Try: "otsu", "triangle", "percentile"
):
    """
    Sobel + Morphology with full ablation control
    Each parameter has strong theoretical impact
    """
    # 1. CLAHE: Local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    img_cl = clahe.apply(img)

    img_cl = cv2.GaussianBlur(img_cl, (5,5), 1.0)

    # 2. Sobel gradients
    sobelx = cv2.Sobel(img_cl, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    sobely = cv2.Sobel(img_cl, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = np.uint8(255 * magnitude / (np.max(magnitude) + 1e-8))

    # 3. Thresholding
    if thresh_method == "otsu":
        _, binary = cv2.threshold(magnitude, 0, 255, cv2.THRESH_OTSU)
    elif thresh_method == "triangle":
        _, binary = cv2.threshold(magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    elif thresh_method == "percentile":
        thresh = np.percentile(magnitude, 92)
        binary = (magnitude > thresh).astype(np.uint8) * 255
    else:
        binary = (magnitude > 70).astype(np.uint8) * 255  # fallback

    # 4. Morphology: Connect fragments
    if closing_radius > 0:
        binary = closing(binary, disk(closing_radius))

    # 5. Remove noise
    binary = remove_small_objects(binary > 0, min_size=min_object_size)
    binary = binary.astype(np.uint8) * 255

    # Visualize
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result[binary > 0] = (255, 192, 203)  # Pink

    return result, binary

def approach6_sobel_morphology_v2(
    img,
    blur=True,           # ← NEW
    bilateral=True,      # ← NEW (best)
    opening=True,        # ← NEW
    blackhat=True,       # ← NEW (optional but strong)
    clahe_clip=3.0,
    clahe_tile=(8,8),
    sobel_ksize=3,
    closing_radius=5,
    min_object_size=80,
    thresh_method="triangle"
):
    """
    Improve to deal with road textures by adding pre-processing and BlackHat
    """
    img_proc = img.copy()

    # 0. Pre-denoising (CRITICAL for asphalt!)
    if bilateral:
        img_proc = cv2.bilateralFilter(img_proc, d=12, sigmaColor=90, sigmaSpace=90)
    elif blur:
        img_proc = cv2.GaussianBlur(img_proc, (5,5), 1.0)
    
    if opening:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        img_proc = cv2.morphologyEx(img_proc, cv2.MORPH_OPEN, kernel, iterations=2)

    # 1. CLAHE
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    img_cl = clahe.apply(img_proc)

    # 2. BlackHat enhancement (optional but powerful)
    if blackhat:
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13,13))
        bh = cv2.morphologyEx(img_cl, cv2.MORPH_BLACKHAT, se)
        img_cl = cv2.add(img_cl, bh)

    # 3. Sobel
    sobelx = cv2.Sobel(img_cl, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    sobely = cv2.Sobel(img_cl, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = np.uint8(255 * magnitude / (np.max(magnitude) + 1e-8))

    # 4. Thresholding — Triangle works better than Otsu on textured roads
    if thresh_method == "triangle":
        _, binary = cv2.threshold(magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    elif thresh_method == "otsu":
        _, binary = cv2.threshold(magnitude, 0, 255, cv2.THRESH_OTSU)
    else:
        thresh = np.percentile(magnitude, 90)
        binary = (magnitude > thresh).astype(np.uint8) * 255
    # 5. Clean up
    if closing_radius > 0:
        binary = closing(binary, disk(closing_radius))
    binary = remove_small_objects(binary > 0, min_size=min_object_size)
    binary = binary.astype(np.uint8) * 255

    # Visualize
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result[binary > 0] = (255, 192, 203)

    return result, binary

def approach6_sobel_morphology_v3(
    img,
    bilateral_d=9,          # Bilateral filter diameter (9, 15, 21)
    bilateral_sigma=75,     # Bilateral sigma (50, 75, 100)
    guided_iter=2,          # Guided filter iterations (1, 2, 3)
    tophat_size=(15, 15),   # Top/Bottom Hat structuring element (11x11, 15x15, 21x21)
    sobel_ksizes=(3, 5),    # Multi-scale Sobel kernels (3, 5, 7)
    closing_radius=5,       # Morphological closing radius (3, 5, 7)
    min_object_size=80,     # Min size for noise removal (50, 80, 120)
    percentile_thresh=90,   # Percentile threshold (85, 90, 95)
    region_grow_factor=0.9  # Region growing factor (0.85, 0.9, 0.95)
):
    """
    Version 3: Enhanced Sobel + Morphology for textured asphalt roads
    - Hybrid denoising (Bilateral + Guided) to remove speckle
    - Top/Bottom Hat to enhance dark linear cracks
    - Multi-scale Sobel for thin/thick cracks
    - Adaptive percentile + region growing for robust thresholding
    - Skeletonization for clean crack outlines
    """
    # 0. Preprocessing: Aggressive denoising for asphalt texture
    img_proc = img.copy().astype(np.float32)
    # Bilateral filter: Removes speckle while preserving edges
    img_proc = cv2.bilateralFilter(img_proc, d=bilateral_d, sigmaColor=bilateral_sigma, sigmaSpace=bilateral_sigma)
    # Guided filter: Further smooths noise, keeps crack boundaries
    img_proc = cv2.GaussianBlur(img_proc, (5, 5), 1.0)
    # img_proc = cv2.ximgproc.guidedFilter(guide=guide, src=img_proc, radius=4, eps=0.01, dDepth=-1, iterations=guided_iter)

    # 1. Contrast Enhancement + Crack Highlighting
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_cl = clahe.apply(np.uint8(img_proc))

    # Top-Hat (bright features) + Bottom-Hat (dark cracks) to suppress texture
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tophat_size)
    tophat = cv2.morphologyEx(img_cl, cv2.MORPH_TOPHAT, se)
    bothat = cv2.morphologyEx(img_cl, cv2.MORPH_BLACKHAT, se)
    enhanced = cv2.addWeighted(img_cl, 1.0, bothat, 0.5, 0.0)  # Boost dark cracks

    # 2. Multi-Scale Sobel Gradients
    mags = []
    for ksize in sobel_ksizes:
        sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=ksize)
        mag = np.sqrt(sobelx**2 + sobely**2)
        mags.append(mag)
    magnitude = np.max(mags, axis=0)  # Combine multi-scale responses
    magnitude = np.uint8(255 * magnitude / (np.max(magnitude) + 1e-8))

    # 3. Adaptive Thresholding with Region Growing
    thresh = np.percentile(magnitude, percentile_thresh)
    binary_init = (magnitude > thresh).astype(np.uint8) * 255

    # Region growing: Expand from initial mask
    marker = np.zeros_like(magnitude, dtype=np.float32)
    marker[binary_init > 0] = magnitude[binary_init > 0] * region_grow_factor
    binary = cv2.dilate(marker.astype(np.uint8), None, iterations=2)
    binary = (binary > 0).astype(np.uint8) * 255

    # 4. Morphology: Connect and refine
    if closing_radius > 0:
        binary = closing(binary, disk(closing_radius))
    binary = skeletonize(binary > 0)  # Thin to 1-pixel width for clean cracks
    binary = remove_small_objects(binary, min_size=min_object_size)
    binary = binary.astype(np.uint8) * 255

    # 5. Visualize
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result[binary > 0] = (255, 192, 203)  # Pink

    return result, binary

def approach7_canny_hysteresis(img):
    """Standard Canny with adaptive thresholds + closing"""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_cl = clahe.apply(img)

    med = np.median(img_cl)
    low = int(max(0, 0.7 * med))
    high = int(min(255, 1.3 * med))

    edges = cv2.Canny(img_cl, low, high, L2gradient=True)
    edges = closing(edges, disk(3))
    edges = remove_small_objects(edges > 0, min_size=50).astype(np.uint8) * 255

    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result[edges > 0] = (0, 165, 255)  # Orange-red

    return result, edges

def approach6_gradient_disruption_v4(
    img,
    blur_sigma=1.0,           # Pre-blur (0.5–2.0)
    clahe_clip=3.0,
    clahe_tile=(8,8),
    ksize=3,                  # Sobel kernel
    disruption_thresh=70,     # Gradient disruption threshold
    magnitude_weight=0.6,     # Balance: magnitude vs disruption
    min_crack_length=100,     # Remove tiny false segments
    closing_radius=7,         # Connect broken parts
    skeleton=True             # Final thin crack lines
):
    """
    Version 4: Gradient Disruption Detector
    Detects cracks as regions of abrupt gradient change (disruption)
    Then connects components into clean crack networks
    """
    # 1. Preprocessing
    img_f = gaussian_filter(img.astype(float), sigma=blur_sigma)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    img_cl = clahe.apply(np.uint8(img_f))

    # 2. Gradient magnitude and direction
    sobelx = cv2.Sobel(img_cl, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(img_cl, cv2.CV_64F, 0, 1, ksize=ksize)
    
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    direction = np.arctan2(sobely, sobelx)  # Angle in radians

    # 3. Gradient of direction = "Disruption" (how much direction changes)
    dir_x = cv2.Sobel(direction, cv2.CV_64F, 1, 0, ksize=3)
    dir_y = cv2.Sobel(direction, cv2.CV_64F, 0, 1, ksize=3)
    disruption = np.sqrt(dir_x**2 + dir_y**2)

    # 4. Normalize both
    mag_norm = magnitude / (magnitude.max() + 1e-8)
    dis_norm = disruption / (disruption.max() + 1e-8)

    # 5. Combine: Crack = High magnitude AND High disruption
    crack_score = magnitude_weight * mag_norm + (1 - magnitude_weight) * dis_norm
    crack_score = np.uint8(255 * crack_score)

    # 6. Threshold + Clean
    thresh = np.percentile(crack_score, disruption_thresh)
    binary = (crack_score > thresh).astype(np.uint8) * 255

    # 7. Connect broken segments
    if closing_radius > 0:
        binary = closing(binary, disk(closing_radius))

    # 8. Remove small noise, keep real cracks
    binary = remove_small_objects(binary > 0, min_size=min_crack_length)
    binary = binary.astype(np.uint8) * 255

    # 9. Optional: Thin to 1-pixel cracks
    if skeleton:
        binary = skeletonize(binary > 0).astype(np.uint8) * 255

    # 10. Visualize
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result[binary > 0] = (255, 105, 180)  # Hot pink for v4

    return result, binary, crack_score  # Return score for visualization

# ================================================================
# MAIN – ONLY APPROACH 6 & 7
# ================================================================
if __name__ == "__main__":
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    image_path = "dataset/Concrete & Pavement Crack Dataset/Positive/00020.jpg"
    image_path = "dataset/evaluation/CrackForest-dataset-master/image/00770.jpg"
    image_path = "example/001.jpg" 
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image not found! Check path.")

    result, mask, score = approach6_gradient_disruption_v4(
        img,
        blur_sigma=12.0,
        disruption_thresh=50,
        magnitude_weight=0.5,
        min_crack_length=120,
        closing_radius=9
    )
    
    cv2.imwrite(f"{output_dir}/v4_result.jpg", result)
    cv2.imwrite(f"{output_dir}/v4_mask.png", mask)
    cv2.imwrite(f"{output_dir}/v4_score.jpg", score)  

    results = []
    masks = []
    names = [
        "Original",
        "6. Sobel+Morphology_v1",
        # "6. Sobel+Morphology_v2",
        # "6. Sobel+Morphology_v3",
        "7. Canny Adaptive",
    ]

    # original
    results.append(img)
    masks.append(None)

    # approach 6
    r6, m6 = approach6_sobel_morphology_v1(
        img,
        clahe_clip=4.0,
        clahe_tile=(10, 10),
        sobel_ksize=5,
        closing_radius=3,
        min_object_size=50,
        thresh_method="otsu"
    )
    results.append(r6)
    masks.append(m6)

    # r6, m6 = approach6_sobel_morphology_v2(
    #     img,
    #     clahe_clip=4.0,
    #     clahe_tile=(10, 10),
    #     sobel_ksize=5,
    #     closing_radius=3,
    #     min_object_size=50,
    #     thresh_method="triangle"
    # )
    # results.append(r6)
    # masks.append(m6)
    
    # r6, m6 = approach6_sobel_morphology_v3(
    #     img,
    #     bilateral_d=9,
    #     bilateral_sigma=75,
    #     guided_iter=2,
    #     tophat_size=(15, 15),
    #     sobel_ksizes=(3, 5),
    #     closing_radius=5,
    #     min_object_size=80,
    #     percentile_thresh=90,
    #     region_grow_factor=0.9
    # )
    # results.append(r6)
    # masks.append(m6)
    
    # approach 7
    r7, m7 = approach7_canny_hysteresis(img)
    results.append(r7)
    masks.append(m7)

    # save masks
    cv2.imwrite(f"{output_dir}/mask_sobel.png", m6)
    cv2.imwrite(f"{output_dir}/mask_canny.png", m7)

    # plotting
    plt.figure(figsize=(20, 10))

    for i in range(len(results)):
        plt.subplot(2, 3, i + 1)
        if i == 0:
            plt.imshow(results[i], cmap="gray")
        else:
            plt.imshow(cv2.cvtColor(results[i], cv2.COLOR_BGR2RGB))
        plt.title(names[i])
        plt.axis("off")

    # show masks
    plt.subplot(2, 3, 4)
    plt.imshow(m6, cmap="gray")
    plt.title("Mask Sobel")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(m7, cmap="gray")
    plt.title("Mask Canny")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_sobel_canny.jpg", dpi=300)
    plt.show()

    print("DONE! Only Approach 6 & 7 executed.")
