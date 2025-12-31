import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
from typing import List, Tuple, Optional
from utils import CrackDectection

class PanoramicCrackInspector:
    def __init__(self):
        self.panorama = None
        self.crack_mask = None
        self.severity_map = None
        self.sift = cv2.SIFT_create()

    def _common_downscale(self, img1, img2, max_dim=1800):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        m = max(h1, w1, h2, w2)

        if m <= max_dim:
            return img1, img2, 1.0

        s = max_dim / float(m)

        new1 = (max(2, int(w1 * s)), max(2, int(h1 * s)))
        new2 = (max(2, int(w2 * s)), max(2, int(h2 * s)))

        img1s = cv2.resize(img1, new1, interpolation=cv2.INTER_AREA)
        img2s = cv2.resize(img2, new2, interpolation=cv2.INTER_AREA)

        return img1s, img2s, s

    def _to_gray_u8(self, img):
        if img.ndim == 3:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            g = img

        if g.dtype != np.uint8:
            g = np.clip(g, 0, 255).astype(np.uint8)

        return g

    def compute_overlap_score(
        self,
        img1,
        img2,
        ratio: float = 0.72,
        min_good: int = 12,
        ransac_thr: float = 3.0,
        max_feat_dim: int = 1800,  
    ) -> Tuple[int, float, Optional[np.ndarray], float]:
        img1s, img2s, s = self._common_downscale(img1, img2, max_dim=max_feat_dim)

        g1 = self._to_gray_u8(img1s)
        g2 = self._to_gray_u8(img2s)

        kp1, des1 = self.sift.detectAndCompute(g1, None)
        kp2, des2 = self.sift.detectAndCompute(g2, None)

        if des1 is None or des2 is None:
            return 0, 0.0, None, 0.0

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=80)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        try:
            matches = flann.knnMatch(des1, des2, k=2)
        except Exception:
            return 0, 0.0, None, 0.0

        good = []
        for mp in matches:
            if len(mp) == 2:
                m, n = mp
                if m.distance < ratio * n.distance:
                    good.append(m)

        good_count = len(good)
        if good_count < min_good:
            return good_count, 0.0, None, 0.0

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        try:
            Hs, inliers = cv2.findHomography(
                dst_pts,
                src_pts,
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=ransac_thr,
                confidence=0.999,
            )
        except Exception:
            Hs, inliers = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, ransac_thr)

        if Hs is None or inliers is None:
            return good_count, 0.0, None, 0.0

        # Convert H from small->full: H_full = S^{-1} * H_small * S
        if s != 1.0:
            S = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float64)
            H = np.linalg.inv(S) @ Hs @ S
        else:
            H = Hs

        inlier_ratio = float(inliers.sum()) / (len(inliers) + 1e-6)
        inlier_count = float(inliers.sum())
        score = inlier_count * (inlier_ratio**2)

        return good_count, float(score), H, float(inlier_ratio)

    def find_optimal_order(self, image_paths: List[str]):
        print("Finding optimal image order for panorama (improved)...")

        images = [cv2.imread(p) for p in image_paths]
        n = len(images)

        if n <= 2:
            return image_paths, images

        score = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            for j in range(i + 1, n):
                good, s, _, inl = self.compute_overlap_score(images[i], images[j])
                score[i, j] = s
                score[j, i] = s
                print(f"Img {i} ↔ Img {j}: good={good}, score={s:.2f}, inlier_ratio={inl:.2f}")

        i0, j0 = np.unravel_index(np.argmax(score), score.shape)

        if score[i0, j0] <= 0:
            print("No strong overlaps detected -> keep original order")
            return image_paths, images

        chain = [i0, j0]
        used = set(chain)
        remaining = [k for k in range(n) if k not in used]

        while remaining:
            best_k = None
            best_side = None
            best_s = -1

            left = chain[0]
            right = chain[-1]

            for k in remaining:
                s_left = score[k, left]
                s_right = score[right, k]

                if s_left > best_s:
                    best_s = s_left
                    best_k = k
                    best_side = "left"

                if s_right > best_s:
                    best_s = s_right
                    best_k = k
                    best_side = "right"

            if best_k is None or best_s <= 0:
                chain.extend(remaining)
                break

            if best_side == "left":
                chain = [best_k] + chain
            else:
                chain = chain + [best_k]

            used.add(best_k)
            remaining.remove(best_k)

        ordered_paths = [image_paths[i] for i in chain]
        ordered_images = [images[i] for i in chain]

        print(f"✓ Optimal order found: {chain}")
        return ordered_paths, ordered_images

    def match_intensity_in_overlap(
        self,
        canvas_u8: np.ndarray,
        warped2_u8: np.ndarray,
        mask1_u8: np.ndarray,
        mask2_u8: np.ndarray,
        robust_clip_percentiles: Tuple[float, float] = (1.0, 99.0),
        min_overlap_pixels: int = 2000,
        max_gain: float = 2.5,
        tile: int = 1024,  # NEW: tile size to avoid OOM
    ) -> np.ndarray:
        overlap = (mask1_u8 > 0) & (mask2_u8 > 0)
        n = int(overlap.sum())

        if n < min_overlap_pixels:
            return warped2_u8

        c = canvas_u8[overlap].astype(np.float32)
        w = warped2_u8[overlap].astype(np.float32)

        lo, hi = robust_clip_percentiles

        c_lo = np.percentile(c, lo, axis=0)
        c_hi = np.percentile(c, hi, axis=0)
        w_lo = np.percentile(w, lo, axis=0)
        w_hi = np.percentile(w, hi, axis=0)

        c_clip = np.clip(c, c_lo, c_hi)
        w_clip = np.clip(w, w_lo, w_hi)

        c_mean = c_clip.mean(axis=0)
        w_mean = w_clip.mean(axis=0)

        c_std = c_clip.std(axis=0) + 1e-6
        w_std = w_clip.std(axis=0) + 1e-6

        gain = c_std / w_std
        gain = np.clip(gain, 1.0 / max_gain, max_gain)
        bias = c_mean - gain * w_mean

        out = warped2_u8.copy()
        H, W = mask2_u8.shape[:2]

        for y0 in range(0, H, tile):
            y1 = min(H, y0 + tile)
            for x0 in range(0, W, tile):
                x1 = min(W, x0 + tile)

                m = mask2_u8[y0:y1, x0:x1] > 0
                if not m.any():
                    continue

                block = out[y0:y1, x0:x1, :]
                b = block.astype(np.float32)

                for ch in range(3):
                    b[..., ch] = gain[ch] * b[..., ch] + bias[ch]

                b = np.clip(b, 0, 255).astype(np.uint8)
                block[m] = b[m]
                out[y0:y1, x0:x1, :] = block

        return out

    def detail_seam_multiband_blend(
        self,
        canvas_u8: np.ndarray,
        warped2_u8: np.ndarray,
        mask1_u8: np.ndarray,
        mask2_u8: np.ndarray,
        num_bands: int = 6,
        seam: str = "gc_colorgrad",
        intensity_match: bool = True,
    ) -> np.ndarray:
        H, W = canvas_u8.shape[:2]
        if intensity_match:
            warped2_u8 = self.match_intensity_in_overlap(
                canvas_u8, warped2_u8, mask1_u8, mask2_u8
            )

        imgs = [canvas_u8, warped2_u8]
        masks = [mask1_u8, mask2_u8]
        corners = [(0, 0), (0, 0)]

        seam_masks = [masks[0].copy(), masks[1].copy()]

        def _try_graphcut_seam(imgs_u8, masks_u8, corners_xy, downscale: float):
            if downscale < 1.0:
                newW = max(64, int(imgs_u8[0].shape[1] * downscale))
                newH = max(64, int(imgs_u8[0].shape[0] * downscale))

                imgs_small = [
                    cv2.resize(imgs_u8[0], (newW, newH), interpolation=cv2.INTER_AREA),
                    cv2.resize(imgs_u8[1], (newW, newH), interpolation=cv2.INTER_AREA),
                ]
                masks_small = [
                    cv2.resize(masks_u8[0], (newW, newH), interpolation=cv2.INTER_NEAREST),
                    cv2.resize(masks_u8[1], (newW, newH), interpolation=cv2.INTER_NEAREST),
                ]
                corners_small = [(0, 0), (0, 0)]
            else:
                imgs_small = [imgs_u8[0], imgs_u8[1]]
                masks_small = [masks_u8[0].copy(), masks_u8[1].copy()]
                corners_small = corners_xy

            if seam == "gc_color":
                seam_finder = cv2.detail_GraphCutSeamFinder("COST_COLOR")
            else:
                seam_finder = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")

            imgs_f = [imgs_small[0].astype(np.float32), imgs_small[1].astype(np.float32)]
            seam_masks_small = [masks_small[0].copy(), masks_small[1].copy()]

            seam_finder.find(imgs_f, corners_small, seam_masks_small)

            if downscale < 1.0:
                seam_masks_big = [
                    cv2.resize(seam_masks_small[0], (W, H), interpolation=cv2.INTER_NEAREST),
                    cv2.resize(seam_masks_small[1], (W, H), interpolation=cv2.INTER_NEAREST),
                ]
                return seam_masks_big

            return seam_masks_small

        max_dim = max(H, W)
        area = H * W

        if max_dim >= 4000 or area >= 16_000_000:
            ds_list = [0.25, 0.20] 
        elif max_dim >= 2500 or area >= 8_000_000:
            ds_list = [0.5, 0.33]  
        else:
            ds_list = [1.0] 

        graphcut_ok = False
        for ds in ds_list:
            try:
                seam_masks = _try_graphcut_seam(imgs, masks, corners, downscale=ds)
                graphcut_ok = True
                break
            except cv2.error:
                graphcut_ok = False
            except Exception:
                graphcut_ok = False

        if not graphcut_ok:
            seam_masks = [masks[0].copy(), masks[1].copy()]
        use_feather = (area >= 20_000_000)

        if use_feather:
            blender = cv2.detail_FeatherBlender()
            blender.setSharpness(0.02)
        else:
            blender = cv2.detail_MultiBandBlender()
            max_bands = max(1, int(np.log2(max(2, min(H, W)))) - 1)
            nb = int(np.clip(num_bands, 1, max_bands))
            blender.setNumBands(nb)

        blender.prepare((0, 0, W, H))

        blender.feed(imgs[0].astype(np.int16), seam_masks[0], (0, 0))
        blender.feed(imgs[1].astype(np.int16), seam_masks[1], (0, 0))

        result_s16, _ = blender.blend(None, None)
        return np.clip(result_s16, 0, 255).astype(np.uint8)

    def stitch_images(self, image_paths, output_path="result/panorama.png"):
        print(f"Stitching {len(image_paths)} images...")

        _, images = self.find_optimal_order(image_paths)

        if len(images) < 2:
            print("Need at least 2 images for stitching")
            self.panorama = images[0] if images else None
            return self.panorama

        result = self.stitch_with_detail_blending(images)

        if result is not None:
            cv2.imwrite(output_path, result)
            self.panorama = result
            return result

        print("Falling back to OpenCV Stitcher...")
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        status, panorama = stitcher.stitch(images)

        if status == cv2.Stitcher_OK:
            cv2.imwrite(output_path, panorama)
            self.panorama = panorama
            return panorama

        print("✗ Stitching failed, using concatenation")
        panorama = np.hstack(images)
        cv2.imwrite(output_path, panorama)
        self.panorama = panorama
        return panorama

    def stitch_with_detail_blending(self, images):
        print(
            "Custom stitching with SIFT + homography + "
            "(seam+multiband + overlap intensity match)..."
        )

        result = images[0]
        for i in range(1, len(images)):
            print(f" Stitching image {i + 1}/{len(images)}...")
            result = self.stitch_pair_detail(result, images[i])

            if result is None:
                print(f" Failed to stitch image {i}")
                return None

        print("✓ Panorama created with detail blending")
        return result

    def stitch_pair_detail(
        self,
        img1,
        img2,
        ratio: float = 0.72,
        min_good: int = 12,
        ransac_thr: float = 3.0,
        num_bands: int = 6,
        seam: str = "gc_colorgrad",
        intensity_match: bool = True,
    ):
        img1s, img2s, s = self._common_downscale(img1, img2, max_dim=1800)

        g1 = self._to_gray_u8(img1s)
        g2 = self._to_gray_u8(img2s)

        kp1, des1 = self.sift.detectAndCompute(g1, None)
        kp2, des2 = self.sift.detectAndCompute(g2, None)

        if des1 is None or des2 is None:
            return None

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=120)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        try:
            matches = flann.knnMatch(des1, des2, k=2)
        except Exception:
            return None

        good = []
        for mp in matches:
            if len(mp) == 2:
                m, n = mp
                if m.distance < ratio * n.distance:
                    good.append(m)

        if len(good) < min_good:
            print(f" Not enough matches: {len(good)}")
            return None

        print(f" Found {len(good)} good matches (scale={s:.3f})")

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        try:
            Hs, _ = cv2.findHomography(
                dst_pts,
                src_pts,
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=ransac_thr,
                confidence=0.999,
            )
        except Exception:
            Hs, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, ransac_thr)

        if Hs is None:
            return None

        if s != 1.0:
            S = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float64)
            H = np.linalg.inv(S) @ Hs @ S
        else:
            H = Hs

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        warped_corners2 = cv2.perspectiveTransform(corners2, H)

        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        all_corners = np.concatenate((corners1, warped_corners2), axis=0)

        x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)

        out_w, out_h = (x_max - x_min), (y_max - y_min)
        output_size = (out_w, out_h)

        warped_img2 = cv2.warpPerspective(img2, T @ H, output_size)

        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        x0, y0 = -x_min, -y_min
        canvas[y0 : y0 + h1, x0 : x0 + w1] = img1

        mask1 = np.zeros((out_h, out_w), dtype=np.uint8)
        mask1[y0 : y0 + h1, x0 : x0 + w1] = 255

        mask2 = (warped_img2.sum(axis=2) > 0).astype(np.uint8) * 255

        overlap = (mask1 > 0) & (mask2 > 0)
        if not overlap.any():
            out = canvas.copy()
            out[mask2 > 0] = warped_img2[mask2 > 0]
            return out

        blended = self.detail_seam_multiband_blend(
            canvas,
            warped_img2,
            mask1,
            mask2,
            num_bands=num_bands,
            seam=seam,
            intensity_match=intensity_match,
        )
        return blended

    def detect_cracks(self, image_input, output_dir, is_path=True):
        crack_detection_obj = CrackDectection()
        _, final_mask = crack_detection_obj.crack_sobel_bothat(image_input, output_dir)
        self.crack_mask = final_mask
        return final_mask

    def calculate_severity(self, crack_mask):
        print("Calculating severity levels...")

        dist_transform = cv2.distanceTransform(crack_mask, cv2.DIST_L2, 5)
        severity = cv2.normalize(
            dist_transform, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        severity = cv2.morphologyEx(severity, cv2.MORPH_CLOSE, kernel)

        self.severity_map = severity
        return severity

    def create_overlay(self, save_path="result/overlay.png"):
        print("Creating severity overlay...")

        if self.panorama is None or self.severity_map is None:
            print("Need panorama and severity map first!")
            return None

        heatmap = cv2.applyColorMap(self.severity_map, cv2.COLORMAP_JET)

        crack_regions = (self.severity_map > 10).astype(np.uint8) * 255
        crack_regions_3ch = cv2.cvtColor(crack_regions, cv2.COLOR_GRAY2BGR)

        overlay = self.panorama.copy()
        alpha = 0.6

        mask_bool = crack_regions_3ch > 0
        overlay[mask_bool] = (
            overlay[mask_bool] * (1 - alpha) + heatmap[mask_bool] * alpha
        ).astype(np.uint8)

        cv2.imwrite(save_path, overlay)
        return overlay

    def generate_report(self, save_path="result/report.png"):
        print("Generating comprehensive report...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # Original panorama
        axes[0, 0].imshow(cv2.cvtColor(self.panorama, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Panoramic Image", fontsize=14, fontweight="bold")
        axes[0, 0].axis("off")

        # Detected cracks
        axes[0, 1].imshow(self.crack_mask, cmap="gray")
        axes[0, 1].set_title("Detected Cracks", fontsize=14, fontweight="bold")
        axes[0, 1].axis("off")

        # Severity heatmap
        axes[1, 0].imshow(self.severity_map, cmap="hot")
        axes[1, 0].set_title("Crack Severity Heatmap", fontsize=14, fontweight="bold")
        axes[1, 0].axis("off")

        # Overlay
        overlay = self.create_overlay("result/temp_overlay.png")
        axes[1, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("Severity Overlay", fontsize=14, fontweight="bold")
        axes[1, 1].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✓ Report saved to {save_path}")
        
    def full_pipeline(self, image_paths, output_dir="result"):
        os.makedirs(output_dir, exist_ok=True)

        print("=" * 60)
        print("PANORAMIC INFRASTRUCTURE CRACK INSPECTOR")
        print("=" * 60)

        # Step 1: Stitch panorama
        print("\n[1/5] Finding optimal order and stitching panorama...")
        panorama = self.stitch_images(image_paths, f"{output_dir}/panorama.png")

        if panorama is None:
            print("Failed to create panorama!")
            return

        # Step 2: Detect cracks
        print("\n[2/5] Detecting cracks on panorama...")
        crack_mask = self.detect_cracks(panorama, output_dir=output_dir, is_path=False)
        cv2.imwrite(f"{output_dir}/cracks_detected.png", crack_mask)

        # Step 3: Calculate severity
        print("\n[3/5] Calculating crack severity levels...")
        severity = self.calculate_severity(crack_mask)
        cv2.imwrite(f"{output_dir}/severity_map.png", severity)

        # Step 4: Create overlay
        print("\n[4/5] Creating color-coded severity overlay...")
        overlay = self.create_overlay(f"{output_dir}/overlay.png")

        # Step 5: Generate report
        print("\n[5/5] Generating comprehensive report...")
        self.generate_report(f"{output_dir}/report.png")

        # Statistics
        total_crack_pixels = np.sum(crack_mask > 0)
        total_pixels = crack_mask.shape[0] * crack_mask.shape[1]
        crack_percentage = (total_crack_pixels / total_pixels) * 100

        print("\n" + "=" * 60)
        print("INSPECTION COMPLETE")
        print("=" * 60)
        print(f"Panorama size: {panorama.shape[1]} × {panorama.shape[0]} pixels")
        print(f"Crack coverage: {crack_percentage:.3f}% of total area")
        print(f"\nAll results saved to '{output_dir}/' directory")
        print("=" * 60)

if __name__ == "__main__":
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
        "dataset/CrackForest-dataset-master/image/013.jpg"
    ]
    
    if len(image_paths) == 1:
        print("Single image mode - processing without stitching...")

        inspector.panorama = cv2.imread(image_paths[0])

        crack_mask = inspector.detect_cracks(image_paths[0], "result")
        os.makedirs("result", exist_ok=True)
        cv2.imwrite("result/cracks_detected.png", crack_mask)

        _ = inspector.calculate_severity(crack_mask)
        _ = inspector.create_overlay("result/overlay.png")

        inspector.generate_report("result/report.png")

        print("\nProcessing complete! Check the 'result/' directory.")
    else:
        inspector.full_pipeline(image_paths, output_dir="result")