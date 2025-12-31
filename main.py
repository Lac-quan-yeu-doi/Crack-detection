import cv2
import os
import shutil
import matplotlib.pyplot as plt
from utils import *

image_path = "example/065.jpg"
image_path = "real_life_image/jpg/6458.jpg"
output_dir = "final_result"

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

image_path = downsample(cv2.imread(image_path), 0.1)

print("Crack Detection...")
crack_detection_obj = CrackDectection()
original, final_mask = crack_detection_obj.crack_hist_clip(image_path, output_dir, low_thresh_ratio=0.1)

# print(f"Org: {type(original)} - Size: {original.shape}")
# print(f"Mask: {type(final_mask)} - Size: {final_mask.shape}")

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
mask_dilated = cv2.dilate(final_mask, kernel, iterations=1)

print("Inpainting...")
inpaint_obj = Inpaint()
inpainted = inpaint_obj.inpaint_with_gradient(
    original,
    mask_dilated,
    window_size=125,
    stride=5,
    priority="horizontal",
    gaussian_ksize=5,
    gaussian_sigma=0.2,
)

# inpainted = inpaint_obj.inpaint_window_sliding_random(
#     original,
#     mask_dilated,
#     step=6,
#     ratio=0.05,
#     max_size=101,
#     neighbor_weight=0.1,
#     gaussian_ksize=15,
#     gaussian_sigma=0.5)

cv2.imwrite(os.path.join(output_dir, "inpaint.jpg"), inpainted)

overlay_org = overlay_mask_on_image(
    original, mask_dilated, color=(0, 0, 255), alpha=0.5  # red in BGR
)
overlay_inp = overlay_mask_on_image(
    inpainted, mask_dilated, color=(0, 0, 255), alpha=0.5  # red in BGR
)

plt.figure(figsize=(18, 6))

plt.subplot(1, 5, 1)
plt.title("Original")
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 5, 2)
plt.title("Crack Mask")
plt.imshow(mask_dilated, cmap="gray")
plt.axis("off")

plt.subplot(1, 5, 3)
plt.title("Mask Overlay")
plt.imshow(cv2.cvtColor(overlay_org, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 5, 4)
plt.title("Inpainting")
plt.imshow(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 5, 5)
plt.title("Overlay Inpainting")
plt.imshow(cv2.cvtColor(overlay_inp, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.savefig(f"{output_dir}/grid.png")
plt.show()

# Blending
inspector = PanoramicCrackInspector()

image_paths = [
    "crack_image_real_time/example_6.jpg",
    "crack_image_real_time/example_7.jpg",
]
image_paths = [
    "real_life_image/jpg/2025_12_27_17_45_IMG_6462.jpg",
    "real_life_image/jpg/2025_12_27_17_45_IMG_6463.jpg",       
]

image_paths = [
    "dataset/CrackForest-dataset-master/image/012.jpg",
    # "dataset/CrackForest-dataset-master/image/013.jpg"
]
output_dir = "test_blending"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    
if len(image_paths) == 1:
    print("Single image mode - processing without stitching...")

    inspector.panorama = cv2.imread(image_paths[0])

    crack_mask = inspector.detect_cracks(image_paths[0], "result")
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(f"{output_dir}/cracks_detected.png", crack_mask)

    _ = inspector.calculate_severity(crack_mask)
    _ = inspector.create_overlay(f"{output_dir}/overlay.png")

    inspector.generate_report(f"{output_dir}/report.png")

    print(f"\nProcessing complete! Check the '{output_dir}' directory.")
else:
    inspector.full_pipeline(image_paths, output_dir=output_dir)