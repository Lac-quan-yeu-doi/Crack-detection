import os
import cv2
import numpy as np
from pathlib import Path

HOME = os.getcwd()
GT_MASK_DIR   = f"{HOME}/CrackForest-dataset-master/groundtruthn_seg_masks"
PRED_MASK_DIR = f"{HOME}/runs/predict-seg/exp4/binary_masks"

def compute_metrics(pred, gt):
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    TP = np.logical_and(pred, gt).sum()
    FP = np.logical_and(pred, ~gt).sum()
    FN = np.logical_and(~pred, gt).sum()
    TN = np.logical_and(~pred, ~gt).sum()

    eps = 1e-8
    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    accuracy  = (TP + TN) / (TP + TN + FP + FN + eps)
    iou       = TP / (TP + FP + FN + eps)

    return precision, recall, f1, accuracy, iou

metrics = []

for fname in os.listdir(GT_MASK_DIR):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    stem = Path(fname).stem
    gt_path   = os.path.join(GT_MASK_DIR, fname)
    pred_path = os.path.join(PRED_MASK_DIR, stem + ".png")

    if not os.path.exists(pred_path):
        print(f"[WARN] Không có pred mask cho {fname}")
        continue

    gt   = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

    # Resize nếu không cùng size
    if gt.shape != pred.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Binary hóa 0/1
    _, gt_bin   = cv2.threshold(gt,   127, 1, cv2.THRESH_BINARY)
    _, pred_bin = cv2.threshold(pred, 127, 1, cv2.THRESH_BINARY)

    m = compute_metrics(pred_bin, gt_bin)
    metrics.append(m)

metrics = np.array(metrics)
P, R, F1, ACC, IOU = metrics.mean(axis=0)
print(f"Precision={P:.4f}, Recall={R:.4f}, F1={F1:.4f}, Accuracy={ACC:.4f}, IoU={IOU:.4f}")
