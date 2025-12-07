import os
import cv2
import numpy as np
from skimage.morphology import closing, disk, remove_small_objects
import matplotlib.pyplot as plt


def approach6_sobel_morphology(
    img,
    clahe_clip=3.0,           # Try: 1.0, 3.0, 5.0, 10.0
    clahe_tile=(8,8),         # Try: (4,4), (8,8), (16,16)
    sobel_ksize=3,            # Try: 3, 5, 7
    closing_radius=3,         # Try: 1, 3, 5, 7
    min_object_size=50,       # Try: 20, 50, 100, 200
    thresh_method="otsu"      # Try: "otsu", "triangle", "percentile"
):
    """
    Sobel + Morphology with full ablation control
    Each parameter has strong theoretical impact
    """
    # 1. CLAHE: Local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    img_cl = clahe.apply(img)

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


# ================================================================
# MAIN – ONLY APPROACH 6 & 7
# ================================================================
if __name__ == "__main__":
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    image_path = "dataset/Concrete & Pavement Crack Dataset/Positive/00020.jpg"
    image_path = "./329.jpg"  # Alternative test image
    image_path = "dataset/evaluation/CrackForest-dataset-master/image/001.jpg"
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image not found! Check path.")

    results = []
    masks = []
    names = [
        "Original",
        "6. Sobel+Morphology",
        "7. Canny Adaptive",
    ]

    # original
    results.append(img)
    masks.append(None)

    # approach 6
    r6, m6 = approach6_sobel_morphology(
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
