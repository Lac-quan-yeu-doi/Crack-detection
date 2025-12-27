import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import skeletonize
from skimage.filters import frangi
from typing import List, Tuple, Optional


class PanoramicCrackInspector:
    def __init__(self):
        self.panorama = None
        self.crack_mask = None
        self.severity_map = None
        self.sift = cv2.SIFT_create()

    # ============================================================
    # 1) Overlap score
    # ============================================================
    def compute_overlap_score(
        self,
        img1,
        img2,
        ratio: float = 0.72,
        min_good: int = 12,
        ransac_thr: float = 3.0,
    ) -> Tuple[int, float, Optional[np.ndarray], float]:
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
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
            H, inliers = cv2.findHomography(
                dst_pts,
                src_pts,
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=ransac_thr,
                confidence=0.999,
            )
        except Exception:
            H, inliers = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, ransac_thr)

        if H is None or inliers is None:
            return good_count, 0.0, None, 0.0

        inlier_ratio = float(inliers.sum()) / (len(inliers) + 1e-6)
        inlier_count = float(inliers.sum())
        score = inlier_count * (inlier_ratio ** 2)
        return good_count, float(score), H, float(inlier_ratio)

    # ============================================================
    # 2) Optimal order
    # ============================================================
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

    # ============================================================
    # 3) Crop black borders
    # ============================================================
    def crop_to_content(self, pano_u8, min_area_ratio=0.01):
        gray = cv2.cvtColor(pano_u8, cv2.COLOR_BGR2GRAY)
        mask = (gray > 0).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return pano_u8

        h, w = pano_u8.shape[:2]
        area_img = h * w
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < area_img * min_area_ratio:
            return pano_u8

        x, y, ww, hh = cv2.boundingRect(c)
        return pano_u8[y:y + hh, x:x + ww]

    # ============================================================
    # 4) Intensity match in overlap (gain+bias)
    # ============================================================
    def match_intensity_in_overlap(
        self,
        canvas_u8: np.ndarray,
        warped2_u8: np.ndarray,
        mask1_u8: np.ndarray,
        mask2_u8: np.ndarray,
        robust_clip_percentiles: Tuple[float, float] = (1.0, 99.0),
        min_overlap_pixels: int = 2000,
        max_gain: float = 2.5,
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

        out = warped2_u8.astype(np.float32)
        for ch in range(3):
            out[..., ch] = gain[ch] * out[..., ch] + bias[ch]

        return np.clip(out, 0, 255).astype(np.uint8)

    # ============================================================
    # 5) Seam + MultiBand blend
    # ============================================================
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
            warped2_u8 = self.match_intensity_in_overlap(canvas_u8, warped2_u8, mask1_u8, mask2_u8)

        corners = [(0, 0), (0, 0)]
        imgs = [canvas_u8.copy(), warped2_u8.copy()]
        masks = [mask1_u8.copy(), mask2_u8.copy()]

        if seam == "gc_color":
            seam_finder = cv2.detail_GraphCutSeamFinder("COST_COLOR")
        else:
            seam_finder = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")

        imgs_f = [imgs[0].astype(np.float32), imgs[1].astype(np.float32)]
        seam_masks = [masks[0].copy(), masks[1].copy()]
        seam_finder.find(imgs_f, corners, seam_masks)

        blender = cv2.detail_MultiBandBlender()
        blender.setNumBands(int(num_bands))
        blender.prepare((0, 0, W, H))

        blender.feed(imgs[0].astype(np.int16), seam_masks[0], (0, 0))
        blender.feed(imgs[1].astype(np.int16), seam_masks[1], (0, 0))

        result_s16, _ = blender.blend(None, None)
        return np.clip(result_s16, 0, 255).astype(np.uint8)

    # ============================================================
    # 6) Stitch
    # ============================================================
    def stitch_images(self, image_paths, output_path="result/panorama.png"):
        print(f"Stitching {len(image_paths)} images...")

        _, images = self.find_optimal_order(image_paths)

        if len(images) < 2:
            print("Need at least 2 images for stitching")
            self.panorama = images[0] if images else None
            return self.panorama

        result = self.stitch_with_detail_blending(images)

        if result is not None:
            result = self.crop_to_content(result)
            cv2.imwrite(output_path, result)
            self.panorama = result
            return result

        print("Falling back to OpenCV Stitcher...")
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        status, panorama = stitcher.stitch(images)

        if status == cv2.Stitcher_OK:
            panorama = self.crop_to_content(panorama)
            cv2.imwrite(output_path, panorama)
            self.panorama = panorama
            return panorama

        print("✗ Stitching failed, using concatenation")
        panorama = np.hstack(images)
        cv2.imwrite(output_path, panorama)
        self.panorama = panorama
        return panorama

    def stitch_with_detail_blending(self, images):
        print("Custom stitching with SIFT + homography + (seam+multiband + overlap intensity match)...")

        result = images[0]
        for i in range(1, len(images)):
            print(f"  Stitching image {i + 1}/{len(images)}...")
            result = self.stitch_pair_detail(result, images[i])
            if result is None:
                print(f"  Failed to stitch image {i}")
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
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
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
            print(f"    Not enough matches: {len(good)}")
            return None

        print(f"    Found {len(good)} good matches")

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        try:
            H, _ = cv2.findHomography(
                dst_pts, src_pts,
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=ransac_thr,
                confidence=0.999
            )
        except Exception:
            H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, ransac_thr)

        if H is None:
            return None

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        warped_corners2 = cv2.perspectiveTransform(corners2, H)

        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        all_corners = np.concatenate((corners1, warped_corners2), axis=0)

        x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        T = np.array([[1, 0, -x_min],
                      [0, 1, -y_min],
                      [0, 0, 1]], dtype=np.float64)

        out_w, out_h = (x_max - x_min), (y_max - y_min)
        output_size = (out_w, out_h)

        warped_img2 = cv2.warpPerspective(img2, T @ H, output_size)

        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        x0, y0 = -x_min, -y_min
        canvas[y0:y0 + h1, x0:x0 + w1] = img1

        mask1 = np.zeros((out_h, out_w), dtype=np.uint8)
        mask1[y0:y0 + h1, x0:x0 + w1] = 255

        mask2 = ((warped_img2.sum(axis=2) > 0).astype(np.uint8) * 255)

        overlap = ((mask1 > 0) & (mask2 > 0))
        if not overlap.any():
            out = canvas.copy()
            out[mask2 > 0] = warped_img2[mask2 > 0]
            return out

        blended = self.detail_seam_multiband_blend(
            canvas, warped_img2, mask1, mask2,
            num_bands=num_bands,
            seam=seam,
            intensity_match=intensity_match
        )
        return blended

    # ============================================================
    # 7) Crack detection (IMPROVED - panorama-robust)
    # ============================================================
    def detect_cracks(self, image_input, is_path=True):
        """
        Your original crack detection algorithm
        """
        if is_path:
            gray = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        else:
            if len(image_input.shape) == 3:
                gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_input
                
        gray_norm = gray.astype(np.float32) / 255.0
        
        # Gaussian blur 
        sigma = 11
        blur = cv2.GaussianBlur(gray_norm, (2*math.ceil(2*sigma)+1,)*2, sigma)
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

        global_threshold_ratio = 3
        threshold = global_threshold_ratio * np.mean(mag)
        mag[mag < threshold] = 0

        # NMS
        def non_max_suppression(data, win):
            data_max = ndimage.maximum_filter(data, footprint=win, mode='constant')
            data_max[data != data_max] = 0
            return data_max

        def orientated_non_max_suppression(mag, ang):
            ang_quant = np.round(ang / (np.pi/4)) % 4

            winE  = np.array([[0,0,0],[1,1,1],[0,0,0]])
            winSE = np.array([[1,0,0],[0,1,0],[0,0,1]])
            winS  = np.array([[0,1,0],[0,1,0],[0,1,0]])
            winSW = np.array([[0,0,1],[0,1,0],[1,0,0]])

            magE  = non_max_suppression(mag, winE)
            magSE = non_max_suppression(mag, winSE)
            magS  = non_max_suppression(mag, winS)
            magSW = non_max_suppression(mag, winSW)

            result = np.zeros_like(mag)
            result[ang_quant == 0] = magE[ang_quant == 0]
            result[ang_quant == 1] = magSE[ang_quant == 1]
            result[ang_quant == 2] = magS[ang_quant == 2]
            result[ang_quant == 3] = magSW[ang_quant == 3]
            return result

        mag_nms = orientated_non_max_suppression(mag, ang)

        high_thresh = 0.5 * mag_nms.max()
        low_thresh  = 0.2 * mag_nms.max()

        high_mask = mag_nms > high_thresh
        low_mask  = mag_nms > low_thresh

        edges = np.zeros_like(mag_nms, dtype=np.uint8)
        edges[high_mask] = 255

        dilate_ksize = 5
        edges = cv2.dilate(edges, np.ones((dilate_ksize, dilate_ksize), np.uint8), iterations=1) 
        edges[low_mask] = np.where(edges[low_mask] > 0, 255, 0)

        close_ksize = 25
        open_ksize = 6
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        proposed_final_c = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
        proposed_final_c_o = cv2.morphologyEx(proposed_final_c, cv2.MORPH_OPEN, kernel_open)
        proposed_final_c_o_c = cv2.morphologyEx(proposed_final_c_o, cv2.MORPH_CLOSE, kernel_close)
        
        self.crack_mask = proposed_final_c_o_c
        return proposed_final_c_o_c
    
    def analyze_with_hough(self, crack_mask):
        """
        Use Hough Transform to detect structural lines and crack alignment
        """
        print("Analyzing structural features with Hough Transform...")
        
        # Probabilistic Hough Line Transform for cracks
        lines = cv2.HoughLinesP(crack_mask, 1, np.pi/180, threshold=50, 
                                minLineLength=30, maxLineGap=10)
        
        # Standard Hough Transform for structural features (on original image)
        gray_pano = cv2.cvtColor(self.panorama, cv2.COLOR_BGR2GRAY)
        edges_struct = cv2.Canny(gray_pano, 50, 150)
        
        structural_lines = cv2.HoughLines(edges_struct, 1, np.pi/180, threshold=200)
        
        return lines, structural_lines
    
    def calculate_severity(self, crack_mask):
        """
        Calculate crack severity based on:
        - Crack width (via distance transform)
        - Crack density (local pixel count)
        - Crack length
        """
        print("Calculating severity levels...")
        
        # Distance transform to get crack width
        dist_transform = cv2.distanceTransform(crack_mask, cv2.DIST_L2, 5)
        
        # Normalize to 0-255 range for visualization
        severity = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply morphological operations to smooth severity map
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        severity = cv2.morphologyEx(severity, cv2.MORPH_CLOSE, kernel)
        
        self.severity_map = severity
        return severity
    
    def create_overlay(self, save_path="result/overlay.png"):
        """
        Create color-coded severity overlay on panorama
        """
        print("Creating severity overlay...")
        
        if self.panorama is None or self.severity_map is None:
            print("Need panorama and severity map first!")
            return None
        
        # Create heatmap colormap
        heatmap = cv2.applyColorMap(self.severity_map, cv2.COLORMAP_JET)
        
        # Create mask where cracks exist
        crack_regions = (self.severity_map > 10).astype(np.uint8) * 255
        crack_regions_3ch = cv2.cvtColor(crack_regions, cv2.COLOR_GRAY2BGR)
        
        # Blend heatmap with original panorama
        overlay = self.panorama.copy()
        alpha = 0.6  # Transparency factor
        
        # Only overlay where cracks exist
        mask_bool = crack_regions_3ch > 0
        overlay[mask_bool] = (overlay[mask_bool] * (1 - alpha) + 
                             heatmap[mask_bool] * alpha).astype(np.uint8)
        
        cv2.imwrite(save_path, overlay)
        return overlay
    
    def visualize_hough_lines(self, crack_lines, structural_lines, save_path="result/hough_analysis.png"):
        """
        Visualize detected Hough lines on panorama
        """
        print("Visualizing Hough line analysis...")
        
        vis = self.panorama.copy()
        
        # Draw crack lines in red
        if crack_lines is not None:
            for line in crack_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Draw structural lines in green
        if structural_lines is not None:
            for line in structural_lines[:50]:  # Limit to top 50
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        cv2.imwrite(save_path, vis)
        return vis
    
    def generate_report(self, crack_lines, structural_lines, save_path="result/report.png"):
        """
        Generate comprehensive visual report
        """
        print("Generating comprehensive report...")
        
        # Create a 2x3 grid of visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Original Panorama
        axes[0, 0].imshow(cv2.cvtColor(self.panorama, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Panorama', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 2. Detected Cracks
        axes[0, 1].imshow(self.crack_mask, cmap='gray')
        axes[0, 1].set_title('Detected Cracks', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # 3. Severity Map
        axes[0, 2].imshow(self.severity_map, cmap='hot')
        axes[0, 2].set_title('Severity Heatmap', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        
        # 4. Crack Lines (Hough)
        crack_vis = self.panorama.copy()
        if crack_lines is not None:
            for line in crack_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(crack_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        axes[1, 0].imshow(cv2.cvtColor(crack_vis, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Crack Lines (Hough)', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # 5. Structural Lines
        struct_vis = self.panorama.copy()
        if structural_lines is not None:
            for line in structural_lines[:50]:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(struct_vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
        axes[1, 1].imshow(cv2.cvtColor(struct_vis, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('Structural Features', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        # 6. Final Overlay
        overlay = self.create_overlay("result/temp_overlay.png")
        axes[1, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title('Severity Overlay', fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Report saved to {save_path}")
    
    def full_pipeline(self, image_paths, output_dir="result"):
        """
        Run the complete panoramic crack inspection pipeline
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 60)
        print("PANORAMIC INFRASTRUCTURE CRACK INSPECTOR")
        print("=" * 60)
        
        # Step 1: Stitch images with optimal ordering
        print("\n[1/6] Finding optimal order and stitching panorama...")
        panorama = self.stitch_images(image_paths, f"{output_dir}/panorama.png")
        
        if panorama is None:
            print("Failed to create panorama!")
            return
        
        # Step 2: Detect cracks
        print("\n[2/6] Detecting cracks on panorama...")
        crack_mask = self.detect_cracks(panorama, is_path=False)
        cv2.imwrite(f"{output_dir}/cracks_detected.png", crack_mask)
        
        # Step 3: Hough analysis
        print("\n[3/6] Analyzing with Hough Transform...")
        crack_lines, structural_lines = self.analyze_with_hough(crack_mask)
        
        # Step 4: Calculate severity
        print("\n[4/6] Calculating severity levels...")
        severity = self.calculate_severity(crack_mask)
        cv2.imwrite(f"{output_dir}/severity_map.png", severity)
        
        # Step 5: Create overlay
        print("\n[5/6] Creating color-coded overlay...")
        overlay = self.create_overlay(f"{output_dir}/overlay.png")
        
        # Step 6: Generate report
        print("\n[6/6] Generating comprehensive report...")
        self.generate_report(crack_lines, structural_lines, f"{output_dir}/report.png")
        
        # Statistics
        total_crack_pixels = np.sum(crack_mask > 0)
        total_pixels = crack_mask.shape[0] * crack_mask.shape[1]
        crack_percentage = (total_crack_pixels / total_pixels) * 100
        
        print("\n" + "=" * 60)
        print("INSPECTION COMPLETE")
        print("=" * 60)
        print(f"Panorama size: {panorama.shape[1]} x {panorama.shape[0]} pixels")
        print(f"Crack coverage: {crack_percentage:.2f}%")
        print(f"Detected crack lines: {len(crack_lines) if crack_lines is not None else 0}")
        print(f"Detected structural lines: {len(structural_lines) if structural_lines is not None else 0}")
        print(f"\nAll results saved to '{output_dir}/' directory")
        print("=" * 60)


# Example usage
if __name__ == "__main__":
    # Initialize inspector
    inspector = PanoramicCrackInspector()
    
    # Example with multiple images for panorama stitching
    # The system will automatically find the best order
    image_paths = [
        "dataset/CrackForest-dataset-master/image/012.jpg",
        "dataset/CrackForest-dataset-master/image/013.jpg",
    ]
    
    # If you only have one image, it will process it directly
    if len(image_paths) == 1:
        print("Single image mode - processing without stitching...")
        inspector.panorama = cv2.imread(image_paths[0])
        crack_mask = inspector.detect_cracks(image_paths[0])
        cv2.imwrite("result/cracks_detected.png", crack_mask)
        
        crack_lines, structural_lines = inspector.analyze_with_hough(crack_mask)
        severity = inspector.calculate_severity(crack_mask)
        overlay = inspector.create_overlay("result/overlay.png")
        inspector.generate_report(crack_lines, structural_lines, "result/report.png")
        
        print("\nProcessing complete! Check the 'result/' directory.")
    else:
        # Run full pipeline with multiple images
        # Images will be automatically ordered for best panorama
        inspector.full_pipeline(image_paths, output_dir="result")
