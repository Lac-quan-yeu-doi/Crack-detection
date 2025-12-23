import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import medial_axis

def crack(image_path):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
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
    cv2.imwrite("result/proposed_final_c_o_c.png", proposed_final_c_o_c)

if __name__ == "__main__":
    image_path = "D:/University/Computer Vision/BTL/example/005.jpg"
    crack(image_path)