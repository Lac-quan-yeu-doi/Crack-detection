import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import medial_axis

class CrackDectection:
    def __init__(self):
        pass
    
    def crack_pure_sobel(
        self,
        image_input,
        output_dir="result",
        ksize=15,
        global_threshold_ratio=3
    ):
        os.makedirs(output_dir, exist_ok=True)

        if isinstance(image_input, str):
            original_color = cv2.imread(image_input)
            gray = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise FileNotFoundError(f"Could not read image from path: {image_input}")
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

        mag_raw = np.hypot(sobelx, sobely) # sobelx^2 + sobely^2

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
        open_ksize=6
    ):
        os.makedirs(output_dir, exist_ok=True)

        if isinstance(image_input, str):
            original_color = cv2.imread(image_input)
            gray = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise FileNotFoundError(f"Could not read image from path: {image_input}")
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

        mag_raw = np.hypot(sobelx, sobely) # sobelx^2 + sobely^2

        # Threshold 
        threshold_raw = global_threshold_ratio * np.mean(mag_raw)
        mag_raw[mag_raw < threshold_raw] = 0

        # Binarize the thresholded magnitude
        mag_raw_binary = np.zeros_like(mag_raw, dtype=np.uint8)
        mag_raw_binary[mag_raw > 0] = 255
        
        cv2.imwrite(os.path.join(output_dir, "sobel_binary_mask.png"), mag_raw_binary)

        mag_close = mag_raw_binary.astype(np.uint8)
        kernel_grad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        mag_close = cv2.morphologyEx(mag_close, cv2.MORPH_CLOSE, kernel_grad)
        
        mag_open = mag_close.astype(np.uint8)
        kernel_grad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        mag_open = cv2.morphologyEx(mag_open, cv2.MORPH_OPEN, kernel_grad)
        
        mag_close = mag_open.astype(np.uint8)
        kernel_grad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        final_mask = cv2.morphologyEx(mag_close, cv2.MORPH_CLOSE, kernel_grad)
        
        cv2.imwrite(os.path.join(output_dir, "sobel_coc_mask.png"), mag_raw_binary)
        
        return original_color, final_mask
        
    def crack_sobel_bothat(
        self,
        image_input,
        output_dir="result",
        ksize=21,
        global_threshold_ratio=3,
        close_ksize=21,
        open_ksize=6
    ):
        os.makedirs(output_dir, exist_ok=True)

        if isinstance(image_input, str):
            original_color = cv2.imread(image_input)
            gray = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise FileNotFoundError(f"Could not read image from path: {image_input}")
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
        
        cv2.imwrite(os.path.join(output_dir, "bothat_sobel_mask.png"), bothat_mag_binary)

        # COC
        mag_close = bothat_mag_binary.astype(np.uint8)
        kernel_grad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        mag_close = cv2.morphologyEx(mag_close, cv2.MORPH_CLOSE, kernel_grad)
        
        mag_open = mag_close.astype(np.uint8)
        kernel_grad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        mag_open = cv2.morphologyEx(mag_open, cv2.MORPH_OPEN, kernel_grad)
        
        mag_close = mag_open.astype(np.uint8)
        kernel_grad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        final_mask = cv2.morphologyEx(mag_close, cv2.MORPH_CLOSE, kernel_grad)
        
        cv2.imwrite(os.path.join(output_dir, "bothat_sobel_coc_mask.png"), final_mask)
        
        return original_color, final_mask
        
    def crack_canny(
        self,
        image_input,
        output_dir="result",       
        gaussian_ksize = 9,
        gaussian_sigma = 1.5,
        sobel_ksize = 9,
        dilate_ksize = 5,
        close_ksize = 15,
        open_ksize = 5
    ):
        os.makedirs(output_dir, exist_ok=True)

        if isinstance(image_input, str):
            original_color = cv2.imread(image_input)
            gray = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise FileNotFoundError(f"Could not read image from path: {image_input}")
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

        blurred = cv2.GaussianBlur(gray_norm, (gaussian_ksize,gaussian_ksize), sigmaX=gaussian_sigma)
        cv2.imwrite(os.path.join(output_dir, "blurred.png"), blurred)
        gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
        gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
        mag = np.hypot(gx, gy)
        ang = np.arctan2(gy, gx)
        cv2.imwrite(os.path.join(output_dir, "blurred_sobel.png"), mag)

        def non_max_suppression_manual(mag, ang):
            ang_quant = np.round(ang / (np.pi/4)) % 4 
            winE  = np.array([[0,0,0],[1,1,1],[0,0,0]])  
            winSE = np.array([[1,0,0],[0,1,0],[0,0,1]])  
            winS  = np.array([[0,1,0],[0,1,0],[0,1,0]]) 
            winSW = np.array([[0,0,1],[0,1,0],[1,0,0]])  

            def nms_dir(data, win):
                data_max = ndimage.maximum_filter(data, footprint=win, mode='constant')
                return np.where(data == data_max, data, 0)

            nms = np.zeros_like(mag)
            nms[ang_quant == 0] = nms_dir(mag, winE)[ang_quant == 0]
            nms[ang_quant == 1] = nms_dir(mag, winSE)[ang_quant == 1]
            nms[ang_quant == 2] = nms_dir(mag, winS)[ang_quant == 2]
            nms[ang_quant == 3] = nms_dir(mag, winSW)[ang_quant == 3]
            
            return nms

        mag_nms = non_max_suppression_manual(mag, ang)

        high_thresh = 0.5 * mag_nms.max()
        low_thresh  = 0.2 * mag_nms.max()

        high_mask = mag_nms > high_thresh
        low_mask  = mag_nms > low_thresh

        edges = np.zeros_like(mag_nms, dtype=np.uint8)
        edges[high_mask] = 255

        edges = cv2.dilate(edges, np.ones((dilate_ksize, dilate_ksize), np.uint8), iterations=1) 
        edges[low_mask] = np.where(edges[low_mask] > 0, 255, 0)

        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
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
                raise FileNotFoundError(f"Could not read image from path: {image_input}")
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
        enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-8)

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
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))

        binary_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)
        final_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)

        cv2.imwrite(os.path.join(output_dir, "binary_crack_map.png"), final_mask)

        return original_color, final_mask

class Inpaint:
    def __init__(self):
        pass

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
    # image_path = "real_life_image/jpg/2025_12_27_17_45_IMG_6463.jpg"
    # image_path = "real_life_image/jpg/2025_12_27_17_47_IMG_6473.jpg"
    # down = downsample(cv2.imread(image_path), 0.6)
    # crack(down)

    image_path = "example/065.jpg"
    image_path = "example/panorama.jpg"
    output_dir = "final_result"
    os.makedirs(output_dir, exist_ok=True)
    
    crack_detection_obj = CrackDectection()
    original, final_mask = crack_detection_obj.crack_hist_clip(image_path, output_dir)

    # print(f"Org: {type(original)} - Size: {original.shape}")
    # print(f"Mask: {type(final_mask)} - Size: {final_mask.shape}")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_dilated = cv2.dilate(final_mask, kernel, iterations=1)

    print("Running your BIT-PLANE REFLECTION INPAINTING...")
    inpaint_obj = Inpaint()
    inpainted = inpaint_obj.inpaint_with_reflection(
        original,
        mask_dilated,
        window_size=125,
        stride=5,
        priority="horizontal",
        gaussian_ksize=5,
        gaussian_sigma=0.2,
    )

    cv2.imwrite(os.path.join(output_dir, "bitplane_result.jpg"), inpainted)

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
