import cv2
import os
import shutil
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import medial_axis
import random
from typing import List, Tuple, Optional

class CrackDectection:
    def __init__(self):
        pass

    def crack_pure_sobel(
        self, image_input, output_dir="result", ksize=15, global_threshold_ratio=3
    ):
        os.makedirs(output_dir, exist_ok=True)

        if isinstance(image_input, str):
            original_color = cv2.imread(image_input)
            gray = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise FileNotFoundError(
                    f"Could not read image from path: {image_input}"
                )
        # If input is image array
        elif isinstance(image_input, np.ndarray):
            original_color = image_input.copy()
            if image_input.ndim == 3:  # Color image (BGR)
                gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
            elif image_input.ndim == 2:  # Already grayscale
                gray = image_input
            else:
                raise ValueError(
                    f"Invalid image array shape: {image_input.shape}. Expected 2D or 3D."
                )
        else:
            raise TypeError(
                f"Unsupported input type: {type(image_input)}. Must be str (path) or np.ndarray (image)."
            )

        gray_norm = gray.astype(np.float32) / 255.0

        sobelx = cv2.Sobel(gray_norm, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(gray_norm, cv2.CV_64F, 0, 1, ksize=ksize)

        mag_raw = np.hypot(sobelx, sobely)  # sobelx^2 + sobely^2

        # Threshold
        threshold_raw = global_threshold_ratio * np.mean(mag_raw)
        mag_raw[mag_raw < threshold_raw] = 0

        # Binarize the thresholded magnitude
        final_mask = np.zeros_like(mag_raw, dtype=np.uint8)
        final_mask[mag_raw > 0] = 255

        cv2.imwrite(os.path.join(output_dir, "sobel_binary_mask.png"), final_mask)

        return original_color, final_mask

    def crack_sobel_coc(
        self,
        image_input,
        output_dir="result",
        ksize=15,
        global_threshold_ratio=3,
        close_ksize=21,
        open_ksize=6,
    ):
        os.makedirs(output_dir, exist_ok=True)

        if isinstance(image_input, str):
            original_color = cv2.imread(image_input)
            gray = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise FileNotFoundError(
                    f"Could not read image from path: {image_input}"
                )
        # If input is image array
        elif isinstance(image_input, np.ndarray):
            original_color = image_input.copy()
            if image_input.ndim == 3:  # Color image (BGR)
                gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
            elif image_input.ndim == 2:  # Already grayscale
                gray = image_input
            else:
                raise ValueError(
                    f"Invalid image array shape: {image_input.shape}. Expected 2D or 3D."
                )
        else:
            raise TypeError(
                f"Unsupported input type: {type(image_input)}. Must be str (path) or np.ndarray (image)."
            )

        gray_norm = gray.astype(np.float32) / 255.0

        sobelx = cv2.Sobel(gray_norm, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(gray_norm, cv2.CV_64F, 0, 1, ksize=ksize)

        mag_raw = np.hypot(sobelx, sobely)  # sobelx^2 + sobely^2

        # Threshold
        threshold_raw = global_threshold_ratio * np.mean(mag_raw)
        mag_raw[mag_raw < threshold_raw] = 0

        # Binarize the thresholded magnitude
        mag_raw_binary = np.zeros_like(mag_raw, dtype=np.uint8)
        mag_raw_binary[mag_raw > 0] = 255

        cv2.imwrite(os.path.join(output_dir, "sobel_binary_mask.png"), mag_raw_binary)

        mag_close = mag_raw_binary.astype(np.uint8)
        kernel_grad = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_ksize, close_ksize)
        )
        mag_close = cv2.morphologyEx(mag_close, cv2.MORPH_CLOSE, kernel_grad)

        mag_open = mag_close.astype(np.uint8)
        kernel_grad = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_ksize, open_ksize)
        )
        mag_open = cv2.morphologyEx(mag_open, cv2.MORPH_OPEN, kernel_grad)

        mag_close = mag_open.astype(np.uint8)
        kernel_grad = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_ksize, close_ksize)
        )
        final_mask = cv2.morphologyEx(mag_close, cv2.MORPH_CLOSE, kernel_grad)

        cv2.imwrite(os.path.join(output_dir, "sobel_coc_mask.png"), final_mask)

        return original_color, final_mask

    def crack_sobel_bothat(
        self,
        image_input,
        output_dir="result",
        ksize=21,
        global_threshold_ratio=3,
        close_ksize=21,
        open_ksize=6,
    ):
        os.makedirs(output_dir, exist_ok=True)

        if isinstance(image_input, str):
            original_color = cv2.imread(image_input)
            gray = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise FileNotFoundError(
                    f"Could not read image from path: {image_input}"
                )
        # If input is image array
        elif isinstance(image_input, np.ndarray):
            original_color = image_input.copy()
            if image_input.ndim == 3:  # Color image (BGR)
                gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
            elif image_input.ndim == 2:  # Already grayscale
                gray = image_input
            else:
                raise ValueError(
                    f"Invalid image array shape: {image_input.shape}. Expected 2D or 3D."
                )
        else:
            raise TypeError(
                f"Unsupported input type: {type(image_input)}. Must be str (path) or np.ndarray (image)."
            )

        # Bottom-hat
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, se)
        bothat = cv2.subtract(closed, gray)

        bothat = bothat.astype(np.float32) / 255.0

        gx = cv2.Sobel(bothat, cv2.CV_64F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(bothat, cv2.CV_64F, 0, 1, ksize=ksize)
        bothat_mag = np.hypot(gx, gy)

        threshold_raw = global_threshold_ratio * np.mean(bothat_mag)

        bothat_mag_binary = np.zeros_like(bothat_mag, dtype=np.uint8)
        bothat_mag_binary[bothat_mag > threshold_raw] = 255

        cv2.imwrite(
            os.path.join(output_dir, "bothat_sobel_mask.png"), bothat_mag_binary
        )

        # COC
        mag_close = bothat_mag_binary.astype(np.uint8)
        kernel_grad = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_ksize, close_ksize)
        )
        mag_close = cv2.morphologyEx(mag_close, cv2.MORPH_CLOSE, kernel_grad)

        mag_open = mag_close.astype(np.uint8)
        kernel_grad = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_ksize, open_ksize)
        )
        mag_open = cv2.morphologyEx(mag_open, cv2.MORPH_OPEN, kernel_grad)

        mag_close = mag_open.astype(np.uint8)
        kernel_grad = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_ksize, close_ksize)
        )
        final_mask = cv2.morphologyEx(mag_close, cv2.MORPH_CLOSE, kernel_grad)

        cv2.imwrite(os.path.join(output_dir, "bothat_sobel_coc_mask.png"), final_mask)

        return original_color, final_mask

    def crack_canny(
        self,
        image_input,
        output_dir="result",
        gaussian_ksize=9,
        gaussian_sigma=1.5,
        sobel_ksize=9,
        dilate_ksize=5,
        close_ksize=15,
        open_ksize=5,
    ):
        os.makedirs(output_dir, exist_ok=True)

        if isinstance(image_input, str):
            original_color = cv2.imread(image_input)
            gray = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise FileNotFoundError(
                    f"Could not read image from path: {image_input}"
                )
        # If input is image array
        elif isinstance(image_input, np.ndarray):
            original_color = image_input.copy()
            if image_input.ndim == 3:  # Color image (BGR)
                gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
            elif image_input.ndim == 2:  # Already grayscale
                gray = image_input
            else:
                raise ValueError(
                    f"Invalid image array shape: {image_input.shape}. Expected 2D or 3D."
                )
        else:
            raise TypeError(
                f"Unsupported input type: {type(image_input)}. Must be str (path) or np.ndarray (image)."
            )

        gray_norm = gray.astype(np.float32) / 255.0

        blurred = cv2.GaussianBlur(
            gray_norm, (gaussian_ksize, gaussian_ksize), sigmaX=gaussian_sigma
        )
        cv2.imwrite(
            os.path.join(output_dir, "blurred.png"), (blurred * 255).astype(np.uint8)
        )
        gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
        gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
        mag = np.hypot(gx, gy)
        ang = np.arctan2(gy, gx)
        cv2.imwrite(
            os.path.join(output_dir, "blurred_sobel.png"),
            cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        )

        def non_max_suppression_manual(mag, ang):
            ang_quant = np.round(ang / (np.pi / 4)) % 4
            winE = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
            winSE = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            winS = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
            winSW = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

            def nms_dir(data, win):
                data_max = ndimage.maximum_filter(data, footprint=win, mode="constant")
                return np.where(data == data_max, data, 0)

            nms = np.zeros_like(mag)
            nms[ang_quant == 0] = nms_dir(mag, winE)[ang_quant == 0]
            nms[ang_quant == 1] = nms_dir(mag, winSE)[ang_quant == 1]
            nms[ang_quant == 2] = nms_dir(mag, winS)[ang_quant == 2]
            nms[ang_quant == 3] = nms_dir(mag, winSW)[ang_quant == 3]

            return nms

        mag_nms = non_max_suppression_manual(mag, ang)

        high_thresh = 0.5 * mag_nms.max()
        low_thresh = 0.2 * mag_nms.max()

        high_mask = mag_nms > high_thresh
        low_mask = mag_nms > low_thresh

        edges = np.zeros_like(mag_nms, dtype=np.uint8)
        edges[high_mask] = 255

        edges = cv2.dilate(
            edges, np.ones((dilate_ksize, dilate_ksize), np.uint8), iterations=1
        )
        edges[low_mask] = np.where(edges[low_mask] > 0, 255, 0)

        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_ksize, close_ksize)
        )
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_ksize, open_ksize)
        )
        manual_canny = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel)
        manual_canny = cv2.morphologyEx(manual_canny, cv2.MORPH_OPEN, open_kernel)
        final_mask = cv2.morphologyEx(manual_canny, cv2.MORPH_CLOSE, close_kernel)

        cv2.imwrite(os.path.join(output_dir, "canny.png"), final_mask)

        return original_color, final_mask

    def crack_hist_clip(
        self,
        image_input,
        output_dir="result",
        sigma=11,
        sobel_ksize=9,
        threshold_ratio=3,
        high_thresh_ratio=0.5,
        low_thresh_ratio=0.5,
        close_ksize=25,
        open_ksize=6,
    ):
        os.makedirs(output_dir, exist_ok=True)

        if isinstance(image_input, str):
            original_color = cv2.imread(image_input)
            gray = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise FileNotFoundError(
                    f"Could not read image from path: {image_input}"
                )
        # If input is image array
        elif isinstance(image_input, np.ndarray):
            original_color = image_input.copy()
            if image_input.ndim == 3:  # Color image (BGR)
                gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
            elif image_input.ndim == 2:  # Already grayscale
                gray = image_input
            else:
                raise ValueError(
                    f"Invalid image array shape: {image_input.shape}. Expected 2D or 3D."
                )
        else:
            raise TypeError(
                f"Unsupported input type: {type(image_input)}. Must be str (path) or np.ndarray (image)."
            )

        gray_norm = gray.astype(np.float32) / 255.0

        # Heavy Gaussian blur and enhancement
        blur = cv2.GaussianBlur(gray_norm, (2 * math.ceil(2 * sigma) + 1,) * 2, sigma)
        enhanced = cv2.subtract(gray_norm, blur)

        # Histogram clipping
        high = np.percentile(enhanced, 50)
        enhanced = np.clip(enhanced, None, high)
        enhanced = (enhanced - enhanced.min()) / (
            enhanced.max() - enhanced.min() + 1e-8
        )

        # Sobel
        sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
        sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
        mag = np.hypot(sobelx, sobely)
        ang = np.arctan2(sobely, sobelx)

        threshold = threshold_ratio * np.mean(mag)
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

        high_thresh = high_thresh_ratio * mag_nms.max()
        low_thresh = low_thresh_ratio * mag_nms.max()

        edges = np.zeros_like(mag_nms, dtype=np.uint8)
        edges[mag_nms > high_thresh] = 255
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
        edges[mag_nms > low_thresh] = np.where(edges[mag_nms > low_thresh] > 0, 255, 0)

        # Morphological operations
        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_ksize, close_ksize)
        )
        kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_ksize, open_ksize)
        )

        binary_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)
        final_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)

        cv2.imwrite(os.path.join(output_dir, "binary_crack_map.png"), final_mask)

        return original_color, final_mask

class Inpaint:
    def __init__(self):
        pass
    
    def inpaint_diffusion(original_img, mask, radius=5, method="telea"):
        flags = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
        return cv2.inpaint(original_img, mask, inpaintRadius=radius, flags=flags)
    
    def inpaint_morphological(original_img, mask, close_ksize=21):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        closed = cv2.morphologyEx(original_img, cv2.MORPH_CLOSE, kernel, iterations=3)

        result = original_img.copy()
        result[mask > 127] = closed[mask > 127]
        return result

    def inpaint_window_sliding_median(
        self,
        original_img,
        mask,
        step=6,
        ratio=0.05,
        max_size=181,
        neighbor_weight=0.6,
        gaussian_ksize=9,
        gaussian_sigma=1.2,
    ):

        img = original_img.copy().astype(np.float32)
        h, w = img.shape[:2]

        if img.ndim == 2:
            img = img[..., np.newaxis]
            was_grayscale = True
            channels = 1
        else:
            was_grayscale = False
            channels = img.shape[2]

        mask_bool = mask > 127
        if mask_bool.ndim == 3:
            mask_bool = mask_bool.squeeze()

        # Known mask contain non-crack pixels
        known_mask = ~mask_bool.copy()

        crack_coords = list(zip(*np.where(mask_bool)))
        if not crack_coords:
            return original_img.astype(np.uint8)

        print(
            f"Median + Weighted Blend inpainting on {len(crack_coords)} pixels (neighbor_weight={neighbor_weight})..."
        )

        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        filled = 0
        for py, px in crack_coords:
            half_size = 0
            candidates_values = []

            while half_size <= (max_size // 2):
                y1 = max(0, py - half_size)
                y2 = min(h, py + half_size + 1)
                x1 = max(0, px - half_size)
                x2 = min(w, px + half_size + 1)

                square_area = (y2 - y1) * (x2 - x1)
                if square_area <= 1:
                    half_size += step
                    continue

                # Collect values from currently known pixels in square
                current_candidates = []
                for ky in range(y1, y2):
                    for kx in range(x1, x2):
                        if known_mask[ky, kx]:
                            current_candidates.append(img[ky, kx])

                if len(current_candidates) >= ratio * square_area:
                    candidates_values = np.array(current_candidates)
                    break

                half_size += step

            # Fallback: use median of all known pixels if no local candidates
            if len(candidates_values) == 0:
                all_known = img[known_mask]
                if len(all_known) > 0:
                    median_value = np.median(all_known, axis=0)
                else:
                    median_value = np.zeros(channels)
            else:
                median_value = np.median(candidates_values, axis=0)

            # Get 8-neighbor values (already known, including previously filled)
            neighbor_values = []
            valid_neighbor_coords = []
            for dy, dx in neighbors:
                ny, nx = py + dy, px + dx
                if 0 <= ny < h and 0 <= nx < w and known_mask[ny, nx]:
                    neighbor_values.append(img[ny, nx])
                    valid_neighbor_coords.append((ny, nx))

            num_neighbors = len(neighbor_values)

            # Weighted blend
            final_value = median_value * (1.0 - neighbor_weight)

            if num_neighbors > 0:
                neighbor_contrib = np.mean(neighbor_values, axis=0)
                final_value += neighbor_contrib * neighbor_weight
            else:
                # Fully rely on median value in the window
                final_value += median_value * neighbor_weight

            # Assign
            img[py, px] = final_value

            # Update known mask
            known_mask[py, px] = True

            filled += 1
            if filled % 1000 == 0:
                print(f"  Progress: {filled}/{len(crack_coords)}")

        # Gaussian smoothing
        if gaussian_ksize > 1:
            print(
                f"Applying final targeted Gaussian (ksize={gaussian_ksize}, sigma={gaussian_sigma})..."
            )
            blurred = cv2.GaussianBlur(
                img, (gaussian_ksize, gaussian_ksize), gaussian_sigma
            )

            if channels == 1:
                crack_mask_3d = mask_bool
            else:
                crack_mask_3d = np.repeat(mask_bool[..., np.newaxis], channels, axis=2)

            img[crack_mask_3d] = blurred[crack_mask_3d]

        result = np.clip(img, 0, 255).astype(np.uint8)
        if was_grayscale:
            result = result.squeeze()

        return result

    def inpaint_window_sliding_random(
        self,
        original_img,
        mask,
        step=5,
        ratio=0.05,
        max_size=151,
        neighbor_weight=0.4,
        gaussian_ksize=9,
        gaussian_sigma=1.2,
        random_seed=50,
    ):

        if random_seed is not None:
            np.random.seed(random_seed)

        img = original_img.copy().astype(np.float32)
        h, w = img.shape[:2]

        if img.ndim == 2:
            img = img[..., np.newaxis]
            was_grayscale = True
            channels = 1
        else:
            was_grayscale = False
            channels = img.shape[2]

        # Original mask
        mask_bool = mask > 127
        if mask_bool.ndim == 3:
            mask_bool = mask_bool.squeeze()

        # Known mask = non-crack
        known_mask = ~mask_bool

        crack_coords = list(zip(*np.where(mask_bool)))
        if not crack_coords:
            return original_img.astype(np.uint8)

        print(
            f"Inpainting on {len(crack_coords)} crack pixels (neighbor weight={neighbor_weight})..."
        )

        # 8-connectivity
        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        filled = 0
        for py, px in crack_coords:
            # Grow square to find candidate known pixels
            half_size = 0
            candidates = []

            while half_size <= (max_size // 2) and not candidates:
                y1 = max(0, py - half_size)
                y2 = min(h, py + half_size + 1)
                x1 = max(0, px - half_size)
                x2 = min(w, px + half_size + 1)

                square_area = (y2 - y1) * (x2 - x1)
                if square_area <= 1:
                    half_size += step
                    continue

                # Collect currently known pixels in square
                for ky in range(y1, y2):
                    for kx in range(x1, x2):
                        if known_mask[ky, kx]:
                            candidates.append((ky, kx))

                if len(candidates) < ratio * square_area:
                    candidates = []

                half_size += step

            # If still no candidates, use any known pixel in image
            if not candidates:
                known_flat = list(zip(*np.where(known_mask)))
                if known_flat:
                    candidates = [random.choice(known_flat)]

            if not candidates:
                filled += 1
                continue

            # Randomly select one main texture pixel
            main_ky, main_kx = random.choice(candidates)
            main_value = img[main_ky, main_kx]

            # Collect 8-neighbors that are already known
            neighbor_values = []
            for dy, dx in neighbors:
                ny, nx = py + dy, px + dx
                if 0 <= ny < h and 0 <= nx < w and known_mask[ny, nx]:
                    neighbor_values.append(img[ny, nx])

            num_neighbors = len(neighbor_values)
            blended = main_value * (1 - neighbor_weight)

            if num_neighbors > 0:
                blended += np.mean(neighbor_values, axis=0) * neighbor_weight
            else:
                blended = main_value

            # Assign
            img[py, px] = blended

            # Update known mask
            known_mask[py, px] = True

            filled += 1
            if filled % 1000 == 0:
                print(f"  Progress: {filled}/{len(crack_coords)}")

        # Gaussian smoothing
        if gaussian_ksize > 1:
            print(
                f"Applying final targeted Gaussian (ksize={gaussian_ksize}, sigma={gaussian_sigma})..."
            )
            blurred = cv2.GaussianBlur(
                img, (gaussian_ksize, gaussian_ksize), gaussian_sigma
            )

            # Mask for original crack pixels
            if channels == 1:
                crack_mask_3d = mask_bool
            else:
                crack_mask_3d = np.repeat(mask_bool[..., np.newaxis], channels, axis=2)

            img[crack_mask_3d] = blurred[crack_mask_3d]

        result = np.clip(img, 0, 255).astype(np.uint8)
        if was_grayscale:
            result = result.squeeze()

        return result

    def inpaint_with_reflection(
        self,
        original_img: np.ndarray,
        mask: np.ndarray,
        window_size=21,
        stride=5,
        priority="horizontal",
        gaussian_ksize=11,
        gaussian_sigma=1.0,
    ):
        if window_size % 2 == 0:
            raise Exception("Window size needs to be an odd number")
        if gaussian_ksize % 2 == 0:
            raise Exception("Gaussian kernel needs to be an odd number")

        img = original_img.copy().astype(np.uint8)
        h, w = img.shape[:2]

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).astype(np.uint8)
            was_grayscale = True
            channels = 3
        else:
            was_grayscale = False
            channels = img.shape[2]

        mask_bool = mask > 127
        if mask_bool.ndim == 3:
            mask_bool = mask_bool.squeeze()

        original_mask_bool = mask_bool.copy()

        half_win = window_size // 2

        directions = (
            ["horizontal", "vertical"]
            if priority == "horizontal"
            else ["vertical", "horizontal"]
        )

        print("Inpainting bitplane...")
        result = np.zeros_like(img)
        for c in range(channels):
            channel = img[:, :, c]

            bit_planes = [(channel & (1 << b)) >> b for b in range(8)]

            processed_planes = []
            for bp in bit_planes:
                bp_processed = bp.copy()
                mask_processed = mask_bool.copy()

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

                                    for dir in directions:
                                        if dir == "horizontal":  # Horizontal reflection
                                            refl_rx = 2 * half_win - rx
                                            refl_ry = ry
                                        else:  # Vertical reflection
                                            refl_rx = rx
                                            refl_ry = 2 * half_win - ry

                                        # Absolute coordinate
                                        source_y = y1 + refl_ry
                                        source_x = x1 + refl_rx

                                        if 0 <= source_y < h and 0 <= source_x < w:
                                            if not mask_processed[source_y, source_x]:
                                                bp_processed[py, px] = bp[
                                                    source_y, source_x
                                                ]
                                                mask_processed[py, px] = False
                                                break

                processed_planes.append(bp_processed)

            # Reconstruct
            reconstructed = np.zeros((h, w), dtype=np.uint8)
            for b in range(8):
                reconstructed |= processed_planes[b] << b

            result[:, :, c] = reconstructed

        # Gaussian smoothing
        if gaussian_ksize > 1:
            print(
                f"Gaussian blur (ksize={gaussian_ksize}, sigma={gaussian_sigma}) to crack regions..."
            )
            blurred = cv2.GaussianBlur(
                result, (gaussian_ksize, gaussian_ksize), gaussian_sigma
            )

            if channels == 1:
                crack_mask_3d = original_mask_bool
            else:
                crack_mask_3d = np.repeat(
                    original_mask_bool[..., np.newaxis], channels, axis=2
                )

            result[crack_mask_3d] = blurred[crack_mask_3d]

        if was_grayscale:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        return result

    def inpaint_with_gradient(
        self,
        original_img: np.ndarray,
        mask: np.ndarray,
        window_size=21,
        stride=5,
        priority="horizontal",
        gaussian_ksize=11,
        gaussian_sigma=1.0,
    ):
        if window_size % 2 == 0:
            raise Exception("Window size needs to be an odd number")
        if gaussian_ksize % 2 == 0:
            raise Exception("Gaussian kernel needs to be an odd number")

        img = original_img.copy().astype(np.uint8)
        h, w = img.shape[:2]

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).astype(np.uint8)
            was_grayscale = True
            channels = 3
        else:
            was_grayscale = False
            channels = img.shape[2]

        mask_bool = mask > 127
        if mask_bool.ndim == 3:
            mask_bool = mask_bool.squeeze()

        original_mask_bool = mask_bool.copy()

        half_win = window_size // 2

        # Precompute gradient magnitude and angle on grayscale version
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if channels == 3 else img.squeeze()
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.hypot(sobelx, sobely)
        grad_ang = np.arctan2(sobely, sobelx)  # -pi to pi

        result = np.zeros_like(img)

        for c in range(channels):
            channel = img[:, :, c]
            bit_planes = [(channel & (1 << b)) >> b for b in range(8)]

            processed_planes = []
            for bp in bit_planes:
                bp_processed = bp.copy()
                mask_processed = mask_bool.copy()

                for y in range(half_win, h - half_win, stride):
                    for x in range(half_win, w - half_win, stride):
                        y1 = y - half_win
                        y2 = y + half_win + 1
                        x1 = x - half_win
                        x2 = x + half_win + 1

                        mask_window = mask_processed[y1:y2, x1:x2]
                        if not np.any(mask_window):
                            continue

                        # Local gradient
                        local_ang = grad_ang[y1:y2, x1:x2]
                        local_mag = grad_mag[y1:y2, x1:x2]

                        # Weighted average angle
                        angles = local_ang.flatten()
                        weights = local_mag.flatten()
                        if np.sum(weights) > 0:
                            mean_angle = np.arctan2(
                                np.sum(weights * np.sin(angles)),
                                np.sum(weights * np.cos(angles)),
                            )
                        else:
                            mean_angle = 0

                        if abs(np.cos(mean_angle)) > abs(np.sin(mean_angle)):
                            # Dominant horizontal gradient → vertical crack → horizontal reflection
                            directions = ["horizontal", "vertical"]
                        elif (np.cos(mean_angle)) < abs(np.sin(mean_angle)):
                            # Dominant vertical gradient → horizontal crack → vertical reflection
                            directions = ["vertical", "horizontal"]
                        else:
                            if priority == "horizontal":
                                directions = ["horizontal", "vertical"]
                            else:
                                directions = ["vertical", "horizontal"]

                        for ry in range(window_size):
                            for rx in range(window_size):
                                if not mask_window[ry, rx]:
                                    continue

                                py = y1 + ry
                                px = x1 + rx

                                filled = False
                                for dir in directions:
                                    if dir == "horizontal":
                                        refl_rx = 2 * half_win - rx
                                        refl_ry = ry
                                    else:
                                        refl_rx = rx
                                        refl_ry = 2 * half_win - ry

                                    source_y = y1 + refl_ry
                                    source_x = x1 + refl_rx

                                    if 0 <= source_y < h and 0 <= source_x < w:
                                        if not mask_processed[source_y, source_x]:
                                            bp_processed[py, px] = bp[
                                                source_y, source_x
                                            ]
                                            mask_processed[py, px] = False
                                            filled = True
                                            break

                processed_planes.append(bp_processed)

            # Reconstruct
            reconstructed = np.zeros((h, w), dtype=np.uint8)
            for b in range(8):
                reconstructed |= processed_planes[b] << b

            result[:, :, c] = reconstructed

        # Gaussian smoothing
        if gaussian_ksize > 1:
            print(f"Gaussian blur (ksize={gaussian_ksize}, sigma={gaussian_sigma})...")
            blurred = cv2.GaussianBlur(
                result, (gaussian_ksize, gaussian_ksize), gaussian_sigma
            )

            crack_mask_3d = (
                original_mask_bool[..., np.newaxis]
                if channels > 1
                else original_mask_bool
            )
            if channels > 1:
                crack_mask_3d = np.repeat(crack_mask_3d, channels, axis=2)

            result[crack_mask_3d] = blurred[crack_mask_3d]

        if was_grayscale:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        return result

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

def overlay_mask_on_image(image, mask, color=(255, 0, 0), alpha=0.5):
    overlay = image.copy()
    mask_bool = mask > 0

    # Apply color only where mask is True
    overlay[mask_bool] = (1 - alpha) * overlay[mask_bool] + alpha * np.array(
        color, dtype=np.uint8
    )

    return overlay.astype(np.uint8)

def downsample(image, ratio, interpolation=cv2.INTER_AREA):
    if ratio <= 0 or ratio > 1:
        raise ValueError("Ratio must be between 0 and 1")

    h, w = image.shape[:2]
    new_w = int(w * ratio)
    new_h = int(h * ratio)

    # Ensure even dimensions (some algorithms prefer it)
    new_w = new_w if new_w % 2 == 0 else new_w + 1
    new_h = new_h if new_h % 2 == 0 else new_h + 1

    downsampled = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    print(f"Downsampled from {w}x{h} → {new_w}x{new_h} (ratio: {ratio:.3f})")
    cv2.imwrite("result/downsampled_image.png", downsampled)
    return downsampled

def upsample(image, original_shape, interpolation=cv2.INTER_CUBIC):
    return cv2.resize(
        image, (original_shape[1], original_shape[0]), interpolation=interpolation
    )

if __name__ == "__main__":

    # Test crack and inpainting
    image_path = "example/065.jpg"
    output_dir = "final_result"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    crack_detection_obj = CrackDectection()
    original, final_mask = crack_detection_obj.crack_sobel_bothat(
        image_path, output_dir
    )

    # print(f"Org: {type(original)} - Size: {original.shape}")
    # print(f"Mask: {type(final_mask)} - Size: {final_mask.shape}")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_dilated = cv2.dilate(final_mask, kernel, iterations=1)

    print("Inpainting")
    inpaint_obj = Inpaint()
    # inpainted = inpaint_obj.inpaint_with_gradient(
    #     original,
    #     mask_dilated,
    #     window_size=125,
    #     stride=5,
    #     priority="horizontal",
    #     gaussian_ksize=5,
    #     gaussian_sigma=0.2,
    # )

    inpainted = inpaint_obj.inpaint_window_sliding_random(
        original,
        mask_dilated,
        step=6,
        ratio=0.05,
        max_size=101,
        neighbor_weight=0.1,
        gaussian_ksize=15,
        gaussian_sigma=0.5,
    )

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
    plt.title("Bit-Plane Inpainting")
    plt.imshow(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 5, 5)
    plt.title("Overlay Bit-Plane Inpainting")
    plt.imshow(cv2.cvtColor(overlay_inp, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/grid.png")
    plt.show()

    # Test blending 
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