import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import medial_axis

class PanoramicCrackInspector:
    def __init__(self):
        self.panorama = None
        self.crack_mask = None
        self.severity_map = None
        self.sift = cv2.SIFT_create()
        
    def compute_overlap_score(self, img1, img2):
        """
        Compute how well two images overlap using SIFT feature matching
        Returns: (number of good matches, match quality score)
        """
        # Detect and compute SIFT features
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None:
            return 0, 0.0
        
        # Match features using FLANN
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        try:
            matches = flann.knnMatch(des1, des2, k=2)
        except:
            return 0, 0.0
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        # Calculate average match distance (lower is better)
        if len(good_matches) > 0:
            avg_distance = np.mean([m.distance for m in good_matches])
            quality_score = len(good_matches) / (avg_distance + 1)
        else:
            quality_score = 0.0
            
        return len(good_matches), quality_score
    
    def find_optimal_order(self, image_paths):
        """
        Find the optimal ordering of images for panorama stitching
        Uses a greedy approach: start with first image, find best neighbor, repeat
        """
        print("Finding optimal image order for panorama...")
        
        images = [cv2.imread(path) for path in image_paths]
        n = len(images)
        
        if n <= 2:
            return image_paths, images
        
        # Compute pairwise overlap scores
        overlap_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                matches, score = self.compute_overlap_score(images[i], images[j])
                overlap_matrix[i, j] = score
                overlap_matrix[j, i] = score
                print(f"  Image {i} ↔ Image {j}: {matches} matches, score: {score:.2f}")
        
        # Greedy ordering: start with image 0, find best chain
        ordered_indices = [0]
        remaining = list(range(1, n))
        
        while remaining:
            current_idx = ordered_indices[-1]
            
            # Find best next image from remaining
            best_score = -1
            best_idx = None
            
            for idx in remaining:
                score = overlap_matrix[current_idx, idx]
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx is not None and best_score > 0:
                ordered_indices.append(best_idx)
                remaining.remove(best_idx)
            else:
                # No good match found, just add remaining in order
                ordered_indices.extend(remaining)
                break
        
        # Reorder images and paths
        ordered_paths = [image_paths[i] for i in ordered_indices]
        ordered_images = [images[i] for i in ordered_indices]
        
        print(f"✓ Optimal order found: {ordered_indices}")
        return ordered_paths, ordered_images
    
    def pyramid_blending(self, img1, img2, mask, levels=6):
        """
        Perform Laplacian pyramid blending for seamless image fusion
        
        Args:
            img1: First image
            img2: Second image  
            mask: Blending mask (1 for img1, 0 for img2)
            levels: Number of pyramid levels
        """
        # Build Gaussian pyramids for both images
        G1 = img1.copy().astype(float)
        G2 = img2.copy().astype(float)
        GM = mask.copy().astype(float)
        
        gp1 = [G1]
        gp2 = [G2]
        gpm = [GM]
        
        for i in range(levels):
            G1 = cv2.pyrDown(G1)
            G2 = cv2.pyrDown(G2)
            GM = cv2.pyrDown(GM)
            gp1.append(G1)
            gp2.append(G2)
            gpm.append(GM)
        
        # Build Laplacian pyramids
        lp1 = [gp1[levels]]
        lp2 = [gp2[levels]]
        
        for i in range(levels, 0, -1):
            size = (gp1[i-1].shape[1], gp1[i-1].shape[0])
            L1 = gp1[i-1] - cv2.pyrUp(gp1[i], dstsize=size)
            L2 = gp2[i-1] - cv2.pyrUp(gp2[i], dstsize=size)
            lp1.append(L1)
            lp2.append(L2)
        
        # Blend Laplacian pyramids
        blended_pyramid = []
        for l1, l2, gm in zip(lp1, lp2, gpm[::-1]):
            # Ensure mask has same number of channels as images
            if len(l1.shape) == 3:
                gm = np.stack([gm] * 3, axis=2)
            blended = l1 * gm + l2 * (1.0 - gm)
            blended_pyramid.append(blended)
        
        # Reconstruct from blended pyramid
        result = blended_pyramid[0]
        for i in range(1, len(blended_pyramid)):
            size = (blended_pyramid[i].shape[1], blended_pyramid[i].shape[0])
            result = cv2.pyrUp(result, dstsize=size) + blended_pyramid[i]
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def stitch_images(self, image_paths, output_path="result/panorama.png"):
        """
        Stitch multiple images into a panorama with optimal ordering
        """
        print(f"Stitching {len(image_paths)} images...")
        
        # Find optimal order
        ordered_paths, images = self.find_optimal_order(image_paths)
        
        if len(images) < 2:
            print("Need at least 2 images for stitching")
            self.panorama = images[0] if images else None
            return self.panorama
        
        # Start stitching with pyramid blending
        result = self.stitch_with_pyramid_blending(images)
        
        if result is not None:
            cv2.imwrite(output_path, result)
            self.panorama = result
            return result
        
        # Fallback to OpenCV stitcher if custom method fails
        print("Falling back to OpenCV Stitcher...")
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        status, panorama = stitcher.stitch(images)
        
        if status == cv2.Stitcher_OK:
            print("✓ Panorama created with OpenCV Stitcher")
            cv2.imwrite(output_path, panorama)
            self.panorama = panorama
            return panorama
        else:
            print(f"✗ Stitching failed, using concatenation")
            panorama = np.hstack(images)
            cv2.imwrite(output_path, panorama)
            self.panorama = panorama
            return panorama
    
    def stitch_with_pyramid_blending(self, images):
        """
        Custom stitching with SIFT matching and pyramid blending
        """
        print("Custom stitching with pyramid blending...")
        
        result = images[0]
        
        for i in range(1, len(images)):
            print(f"  Stitching image {i+1}/{len(images)}...")
            result = self.stitch_pair_pyramid(result, images[i])
            
            if result is None:
                print(f"  Failed to stitch image {i}")
                return None
        
        print("✓ Panorama created with pyramid blending")
        return result
    
    def stitch_pair_pyramid(self, img1, img2):
        """
        Stitch two images using SIFT + homography + pyramid blending
        """
        # Detect and compute SIFT features
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None:
            return None
        
        # Match features
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        try:
            matches = flann.knnMatch(des1, des2, k=2)
        except:
            return None
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            print(f"    Not enough matches: {len(good_matches)}")
            return None
        
        print(f"    Found {len(good_matches)} good matches")
        
        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            return None
        
        # Calculate canvas size
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        corners = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, H)
        
        all_corners = np.concatenate((
            np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2),
            warped_corners
        ), axis=0)
        
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        
        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
        
        # Warp image 2
        output_size = (x_max - x_min, y_max - y_min)
        warped_img2 = cv2.warpPerspective(img2, translation @ H, output_size)
        
        # Create canvas for image 1
        canvas = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        canvas[-y_min:-y_min+h1, -x_min:-x_min+w1] = img1
        
        # Create blending mask
        mask1 = np.zeros((output_size[1], output_size[0]), dtype=np.float32)
        mask1[-y_min:-y_min+h1, -x_min:-x_min+w1] = 1.0
        
        mask2 = (warped_img2.sum(axis=2) > 0).astype(np.float32)
        
        # Find overlap region
        overlap = (mask1 > 0) & (mask2 > 0)
        
        if not overlap.any():
            # No overlap, simple combination
            result = canvas.copy()
            result[mask2 > 0] = warped_img2[mask2 > 0]
            return result
        
        # Create smooth transition mask for overlap region
        # Use distance transform for smooth blending
        overlap_mask1 = mask1 * overlap
        overlap_mask2 = mask2 * overlap
        
        dist1 = cv2.distanceTransform((overlap_mask1 > 0).astype(np.uint8), cv2.DIST_L2, 5)
        dist2 = cv2.distanceTransform((overlap_mask2 > 0).astype(np.uint8), cv2.DIST_L2, 5)
        
        # Normalize distances to create blend weights
        total_dist = dist1 + dist2 + 1e-6
        blend_mask = dist1 / total_dist
        
        # Apply pyramid blending in overlap region
        overlap_region = overlap.astype(np.uint8) * 255
        
        # Extract overlap regions from both images
        y_coords, x_coords = np.where(overlap)
        if len(y_coords) == 0:
            result = canvas.copy()
            result[mask2 > 0] = warped_img2[mask2 > 0]
            return result
        
        y_min_overlap, y_max_overlap = y_coords.min(), y_coords.max()
        x_min_overlap, x_max_overlap = x_coords.min(), x_coords.max()
        
        overlap_img1 = canvas[y_min_overlap:y_max_overlap+1, x_min_overlap:x_max_overlap+1]
        overlap_img2 = warped_img2[y_min_overlap:y_max_overlap+1, x_min_overlap:x_max_overlap+1]
        overlap_mask_crop = blend_mask[y_min_overlap:y_max_overlap+1, x_min_overlap:x_max_overlap+1]
        
        # Pyramid blend the overlap region
        if overlap_img1.size > 0 and overlap_img2.size > 0:
            blended_overlap = self.pyramid_blending(overlap_img1, overlap_img2, overlap_mask_crop, levels=4)
            
            # Combine: canvas + blended_overlap + warped_img2 (non-overlap)
            result = canvas.copy()
            result[y_min_overlap:y_max_overlap+1, x_min_overlap:x_max_overlap+1] = blended_overlap
            
            # Add non-overlapping parts of img2
            non_overlap_mask2 = (mask2 > 0) & ~overlap
            result[non_overlap_mask2] = warped_img2[non_overlap_mask2]
        else:
            result = canvas.copy()
            result[mask2 > 0] = warped_img2[mask2 > 0]
        
        return result
    
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
        "D:/University/Computer Vision/BTL/example/065.jpg",
        "example/001.jpg",
        "example/010.jpg"
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