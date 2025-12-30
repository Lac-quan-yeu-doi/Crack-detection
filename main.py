import cv2
import os
import shutil
import matplotlib.pyplot as plt
from utils import *

image_path = "example/065.jpg"
output_dir = "final_result"

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

print("Crack Detection...")
crack_detection_obj = CrackDectection()
original, final_mask = crack_detection_obj.crack_sobel_bothat(image_path, output_dir)

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
