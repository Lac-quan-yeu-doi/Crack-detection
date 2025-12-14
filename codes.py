import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.fft import fft2, ifft2, fftshift

from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes
from skimage.measure import label, regionprops


# -----------------------------
# Phase congruency (phasepack preferred)
# -----------------------------
USE_PHASEPACK = False
try:
    from phasepack import phasecong  # type: ignore
    USE_PHASEPACK = True
except Exception:
    USE_PHASEPACK = False


def _lowpass_filter(shape, cutoff=0.45, sharpness=15):
    rows, cols = shape
    y = np.linspace(-0.5, 0.5, rows, endpoint=False)
    x = np.linspace(-0.5, 0.5, cols, endpoint=False)
    X, Y = np.meshgrid(x, y)
    radius = np.sqrt(X * X + Y * Y)
    filt = 1.0 / (1.0 + (radius / (cutoff + 1e-9)) ** (2 * sharpness))
    return fftshift(filt)


def _log_gabor(shape, wavelength, sigmaOnf=0.55):
    rows, cols = shape
    y = np.linspace(-0.5, 0.5, rows, endpoint=False)
    x = np.linspace(-0.5, 0.5, cols, endpoint=False)
    X, Y = np.meshgrid(x, y)
    radius = np.sqrt(X * X + Y * Y)
    radius[0, 0] = 1e-9
    fo = 1.0 / float(wavelength)
    log_rad = np.log(radius / fo)
    log_gabor = np.exp(-(log_rad * log_rad) / (2.0 * (np.log(sigmaOnf) ** 2 + 1e-9)))
    log_gabor[0, 0] = 0
    return fftshift(log_gabor).astype(np.float32)


def _angular_filter(shape, angle, thetaSigma=1.2):
    rows, cols = shape
    y = np.linspace(-0.5, 0.5, rows, endpoint=False)
    x = np.linspace(-0.5, 0.5, cols, endpoint=False)
    X, Y = np.meshgrid(x, y)
    theta = np.arctan2(-Y, X)
    ds = np.sin(theta) * np.cos(angle) - np.cos(theta) * np.sin(angle)
    dc = np.cos(theta) * np.cos(angle) + np.sin(theta) * np.sin(angle)
    dtheta = np.abs(np.arctan2(ds, dc))
    return fftshift(np.exp(-(dtheta * dtheta) / (2.0 * thetaSigma * thetaSigma))).astype(np.float32)


def phase_congruency_2d(gray, nscale=4, norient=6, minWavelength=3.0, mult=2.1,
                       sigmaOnf=0.55, k=2.0, noiseMethod="median"):
    """
    Simplified Kovesi-like 2D phase congruency fallback.
    Returns: pc (0..1), orient (radians)
    """
    img = gray.astype(np.float32)
    fftim = fft2(img)
    lp = _lowpass_filter(img.shape, cutoff=0.45, sharpness=15).astype(np.float32)
    eps = 1e-9

    totalEnergy = np.zeros_like(img, np.float32)
    totalAn = np.zeros_like(img, np.float32)

    sumE = np.zeros_like(img, np.float32)
    sumO = np.zeros_like(img, np.float32)

    for o in range(norient):
        angl = o * np.pi / norient
        angFilter = _angular_filter(img.shape, angl, thetaSigma=1.2)

        sumAn_or = np.zeros_like(img, np.float32)
        sumE_or = np.zeros_like(img, np.float32)
        sumO_or = np.zeros_like(img, np.float32)

        # noise threshold (estimated at smallest scale)
        tau = None

        for s in range(nscale):
            wavelength = minWavelength * (mult ** s)
            logG = _log_gabor(img.shape, wavelength=wavelength, sigmaOnf=sigmaOnf)
            filt = logG * lp * angFilter

            EO = ifft2(fftim * filt)
            E = np.real(EO).astype(np.float32)
            O = np.imag(EO).astype(np.float32)

            An = np.sqrt(E * E + O * O).astype(np.float32)

            if s == 0:
                # estimate noise (very rough)
                if noiseMethod == "median":
                    tau = np.median(An) / math.sqrt(math.log(4.0) + eps)
                else:
                    tau = np.mean(An)

            sumAn_or += An
            sumE_or += E
            sumO_or += O

        # Energy
        XEnergy = np.sqrt(sumE_or * sumE_or + sumO_or * sumO_or) + eps
        meanE = sumE_or / XEnergy
        meanO = sumO_or / XEnergy

        Energy = sumE_or * meanE + sumO_or * meanO - np.abs(sumE_or * meanO - sumO_or * meanE)

        # Noise threshold T
        if tau is None:
            tau = np.median(sumAn_or) / math.sqrt(math.log(4.0) + eps)
        T = (tau * k)

        energy = np.maximum(Energy - T, 0)

        totalEnergy += energy
        totalAn += sumAn_or

        sumE += energy * np.cos(angl)
        sumO += energy * np.sin(angl)

    pc = totalEnergy / (totalAn + eps)
    pc = np.clip(pc, 0, 1).astype(np.float32)

    orient = np.arctan2(sumO, sumE).astype(np.float32)
    orient = ((orient + np.pi / 2) % np.pi) - np.pi / 2
    orient = (orient % np.pi).astype(np.float32)
    return pc, orient


def compute_pc(gray01, nscale=4, norient=6, minWavelength=3.0, mult=2.1,
               sigmaOnf=0.55, k=2.0, cutOff=0.5, g=10):
    if USE_PHASEPACK:
        out = phasecong(
            gray01, nscale=nscale, norient=norient,
            minWaveLength=minWavelength, mult=mult,
            sigmaOnf=sigmaOnf, k=k, cutOff=cutOff, g=g
        )
        pc = np.clip(out[0], 0, 1).astype(np.float32)
        orient = (out[1] % np.pi).astype(np.float32)
        return pc, orient

    # fallback
    pc, orient = phase_congruency_2d(
        gray01, nscale=nscale, norient=norient,
        minWavelength=minWavelength, mult=mult, sigmaOnf=sigmaOnf, k=k
    )
    return pc, orient


# -----------------------------
# Helpers from notebook flow
# -----------------------------
def norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def orientation_coherence(theta: np.ndarray, win: int = 17) -> np.ndarray:
    z = np.exp(1j * 2.0 * theta)
    real = ndi.uniform_filter(np.real(z), size=win)
    imag = ndi.uniform_filter(np.imag(z), size=win)
    coh = np.sqrt(real * real + imag * imag)
    return np.clip(coh, 0, 1).astype(np.float32)


def hessian_eigs(gray: np.ndarray, sigma: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    I = gray.astype(np.float32)
    I = ndi.gaussian_filter(I, sigma=sigma)
    Ixx = ndi.gaussian_filter(I, sigma=sigma, order=(2, 0))
    Iyy = ndi.gaussian_filter(I, sigma=sigma, order=(0, 2))
    Ixy = ndi.gaussian_filter(I, sigma=sigma, order=(1, 1))

    tmp = np.sqrt((Ixx - Iyy) ** 2 + 4.0 * (Ixy ** 2))
    l1 = 0.5 * (Ixx + Iyy + tmp)
    l2 = 0.5 * (Ixx + Iyy - tmp)

    swap = np.abs(l1) > np.abs(l2)
    l1s = l1.copy()
    l2s = l2.copy()
    l1s[swap], l2s[swap] = l2[swap], l1[swap]
    return l1s.astype(np.float32), l2s.astype(np.float32)


def frangi_like_vesselness(gray_ridge: np.ndarray,
                           sigmas: Tuple[float, ...] = (1.2, 1.8, 2.6, 3.6),
                           beta: float = 0.5,
                           c: float = 10.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Line-likeness for dark cracks (on black-hat map).
    Returns: vessel (0..1), best_sigma_idx, l1_best, l2_best
    """
    eps = 1e-9
    vessel_stack: List[np.ndarray] = []
    l1_stack: List[np.ndarray] = []
    l2_stack: List[np.ndarray] = []

    for s in sigmas:
        l1, l2 = hessian_eigs(gray_ridge, sigma=float(s))
        l1_stack.append(l1)
        l2_stack.append(l2)

        cond = (l2 < 0)  # dark ridges
        Rb = (np.abs(l1) / (np.abs(l2) + eps))
        S = np.sqrt(l1 * l1 + l2 * l2)

        V = np.exp(-(Rb * Rb) / (2.0 * beta * beta)) * (1.0 - np.exp(-(S * S) / (2.0 * c * c)))
        V = V * cond.astype(np.float32)

        V = V / (V.max() + eps)  # per-scale normalization
        vessel_stack.append(V.astype(np.float32))

    vessel_stack_np = np.stack(vessel_stack, axis=0)
    best_idx = np.argmax(vessel_stack_np, axis=0)
    vessel = np.max(vessel_stack_np, axis=0).astype(np.float32)

    l1_stack_np = np.stack(l1_stack, axis=0)
    l2_stack_np = np.stack(l2_stack, axis=0)
    H, W = vessel.shape
    rr = np.arange(H)[:, None]
    cc = np.arange(W)[None, :]
    l1_best = l1_stack_np[best_idx, rr, cc]
    l2_best = l2_stack_np[best_idx, rr, cc]

    return np.clip(vessel, 0, 1), best_idx.astype(np.int32), l1_best.astype(np.float32), l2_best.astype(np.float32)


def line_kernel(length: int = 11, angle_deg: float = 0.0) -> np.ndarray:
    k = np.zeros((length, length), np.uint8)
    c = length // 2
    a = np.deg2rad(angle_deg)
    x0 = int(c - c * np.cos(a))
    y0 = int(c - c * np.sin(a))
    x1 = int(c + c * np.cos(a))
    y1 = int(c + c * np.sin(a))
    cv2.line(k, (x0, y0), (x1, y1), 1, 1)
    return k


def direction_aware_close(mask: np.ndarray, orient: np.ndarray, bins: int = 16, length: int = 17) -> np.ndarray:
    """
    Close along local crack direction (perpendicular to PC orientation).
    """
    out = np.zeros_like(mask, dtype=bool)
    theta = (orient + np.pi / 2) % np.pi
    edges = np.linspace(0, np.pi, bins + 1)
    for i in range(bins):
        t0, t1 = edges[i], edges[i + 1]
        sel = (theta >= t0) & (theta < t1) & mask
        if sel.sum() == 0:
            continue
        ang_deg = float(np.rad2deg(0.5 * (t0 + t1)))
        k = line_kernel(length=length, angle_deg=ang_deg)
        sel_u8 = (sel.astype(np.uint8) * 255)
        closed = cv2.morphologyEx(sel_u8, cv2.MORPH_CLOSE, k)
        out |= (closed > 0)
    return out


def prune_by_size(bin_mask: np.ndarray, min_pixels: int = 70) -> np.ndarray:
    lab = label(bin_mask, connectivity=2)
    out = np.zeros_like(bin_mask, dtype=bool)
    for r in regionprops(lab):
        if r.area >= min_pixels:
            out[lab == r.label] = True
    return out


def prune_skeleton_components(skel_bin: np.ndarray, min_len: int = 140) -> np.ndarray:
    lab = label(skel_bin, connectivity=2)
    out = np.zeros_like(skel_bin, dtype=bool)
    for r in regionprops(lab):
        if r.area >= min_len:
            out[lab == r.label] = True
    return out


def oriented_bridge(skel: np.ndarray, orient: np.ndarray, support_mask: np.ndarray,
                    bins: int = 12, length: int = 19) -> np.ndarray:
    theta = (orient + np.pi / 2) % np.pi
    out = np.zeros_like(skel, dtype=bool)
    edges = np.linspace(0, np.pi, bins + 1)
    for i in range(bins):
        t0, t1 = edges[i], edges[i + 1]
        sel = (theta >= t0) & (theta < t1) & skel
        if sel.sum() == 0:
            continue
        ang_deg = float(np.rad2deg(0.5 * (t0 + t1)))
        k = line_kernel(length=length, angle_deg=ang_deg)
        sel_u8 = sel.astype(np.uint8) * 255
        dil = cv2.dilate(sel_u8, k, iterations=1) > 0
        out |= dil
    out = out & support_mask
    out = skeletonize(out)
    out = prune_by_size(out, min_pixels=70)
    return out


# -----------------------------
# Full flow: image -> FinalMask
# -----------------------------
def predict_final_mask_v4(img_bgr: np.ndarray,
                          *,
                          p_low: float = 97.0,
                          p_high: float = 99.5,
                          blur_sigma: float = 12.0,
                          bh_ksize: int = 15,
                          ridge_sigmas: Tuple[float, ...] = (1.2, 1.8, 2.6, 3.6),
                          vessel_beta: float = 0.5,
                          vessel_c: float = 10.0,
                          pc2_nscale: int = 5,
                          pc2_norient: int = 8,
                          skel_min_len: int = 140,
                          bridge_bins: int = 12,
                          bridge_len: int = 19,
                          pred_dilate_px: int = 2,
                          support_pctl: float = 93.0) -> np.ndarray:
    # gray in [0,1]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # illumination flatten (high-pass)
    lowfreq = cv2.GaussianBlur(gray, (0, 0), float(blur_sigma))
    gray_flat = np.clip(gray - lowfreq + 0.5, 0, 1).astype(np.float32)

    # black-hat (enhance dark thin lines)
    g8 = (gray_flat * 255).astype(np.uint8)
    kbh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(bh_ksize), int(bh_ksize)))
    blackhat = cv2.morphologyEx(g8, cv2.MORPH_BLACKHAT, kbh).astype(np.float32) / 255.0

    gray_pc = norm01(gray_flat)
    gray_ridge = norm01(blackhat)

    # PC tuned
    pc2, orient2 = compute_pc(gray_pc, nscale=pc2_nscale, norient=pc2_norient,
                             minWavelength=3.0, mult=2.0, sigmaOnf=0.55, k=2.5)

    coh2 = orientation_coherence(orient2, win=17)
    pc3 = np.clip(pc2 * (0.35 + 0.65 * coh2), 0, 1).astype(np.float32)

    # vesselness
    vessel, best_idx, l1, l2 = frangi_like_vesselness(gray_ridge, sigmas=ridge_sigmas,
                                                      beta=vessel_beta, c=vessel_c)

    # score
    score1 = pc3 * vessel
    score1 = score1 / (score1.max() + 1e-9)

    anis = (np.abs(l1) - np.abs(l2)) / (np.abs(l1) + np.abs(l2) + 1e-9)
    anis = np.clip(anis, 0, 1).astype(np.float32)

    score2 = score1 * (0.3 + 0.7 * anis)
    score2 = score2 / (score2.max() + 1e-9)

    # weak darkness prior
    dark = np.clip(0.55 - gray_flat, 0, 1).astype(np.float32)
    dark = dark / (dark.max() + 1e-9)

    score3 = score2 * (0.55 + 0.45 * dark)
    score3 = score3 / (score3.max() + 1e-9)

    # smoothing
    score3_s = cv2.GaussianBlur(score3, (0, 0), 1.0)

    # hysteresis thresholds by percentiles (your tuned params)
    t_hi = np.percentile(score3_s, p_high)
    t_lo = np.percentile(score3_s, p_low)
    mask1 = apply_hysteresis_threshold(score3_s, t_lo, t_hi)

    mask1 = ndi.binary_opening(mask1, structure=np.ones((3, 3), np.uint8))
    mask1 = remove_small_objects(mask1, min_size=80)
    mask1 = remove_small_holes(mask1, area_threshold=120)

    # direction-aware connect/close
    mask2 = direction_aware_close(mask1, orient2, bins=16, length=17)

    # line-like component filtering
    lab = label(mask2, connectivity=2)
    keep = np.zeros_like(mask2, dtype=bool)
    for r in regionprops(lab):
        if r.area < 120:
            continue
        if (r.major_axis_length < 35) and (r.eccentricity < 0.90):
            continue
        keep[lab == r.label] = True

    mask2 = keep
    mask2 = remove_small_holes(mask2, area_threshold=200)

    # skeleton & prune
    skel = skeletonize(mask2)
    skel_p = prune_skeleton_components(skel, min_len=skel_min_len)

    # bridge
    final_skel = oriented_bridge(skel_p, orient2, mask2, bins=bridge_bins, length=bridge_len)

    # support clip + thickness
    support_th = np.percentile(score3_s, support_pctl)
    support = (score3_s >= support_th)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pred_dilate_px + 1, 2 * pred_dilate_px + 1))
    thick = cv2.dilate((final_skel.astype(np.uint8) * 255), k, iterations=1) > 0
    final_mask = thick & support

    # cleanup
    final_mask = ndi.binary_closing(final_mask, structure=np.ones((3, 3), np.uint8))
    final_mask = remove_small_objects(final_mask, min_size=200)
    final_mask = remove_small_holes(final_mask, area_threshold=300)

    return (final_mask.astype(np.uint8) * 255)


# -----------------------------
# Dataset IO + metrics
# -----------------------------
def find_default_dirs(base: Path) -> Tuple[Path, Path]:
    candidates_img = [
        base / "image", base / "images", base / "Image", base / "Images",
    ]
    candidates_gt = [
        base / "groundtruth_seg_masks",
        base / "groundtruth", base / "mask", base / "masks",
        base / "groundtruthn_seg_masks",
    ]
    img_dir = next((p for p in candidates_img if p.exists()), None)
    gt_dir = next((p for p in candidates_gt if p.exists()), None)
    if img_dir is None or gt_dir is None:
        raise FileNotFoundError(
            f"Could not auto-detect image/gt folders under: {base}\n"
            "Please pass --img_dir and --gt_dir explicitly."
        )
    return img_dir, gt_dir


def list_images(img_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [p for p in img_dir.rglob("*") if p.suffix.lower() in exts]
    files.sort()
    if not files:
        raise FileNotFoundError(f"No images found in {img_dir}")
    return files


def load_gt_mask(gt_dir: Path, stem: str) -> np.ndarray:
    candidates = [
        gt_dir / f"{stem}.png",
        gt_dir / f"{stem}.jpg",
        gt_dir / f"{stem}.bmp",
        gt_dir / f"{stem}_mask.png",
        gt_dir / f"{stem}_gt.png",
    ]
    gt_path = next((p for p in candidates if p.exists()), None)
    if gt_path is None:
        matches = list(gt_dir.rglob(f"{stem}.*"))
        gt_path = matches[0] if matches else None
    if gt_path is None:
        raise FileNotFoundError(f"GT not found for {stem} under {gt_dir}")

    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise RuntimeError(f"Failed to read GT: {gt_path}")
    return (gt > 0).astype(np.uint8) * 255


def compute_metrics(pred_u8: np.ndarray, gt_u8: np.ndarray) -> Dict[str, float]:
    pred = (pred_u8 > 0)
    gt = (gt_u8 > 0)

    TP = np.logical_and(pred, gt).sum()
    FP = np.logical_and(pred, ~gt).sum()
    FN = np.logical_and(~pred, gt).sum()
    TN = np.logical_and(~pred, ~gt).sum()

    eps = 1e-8
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
    iou = TP / (TP + FP + FN + eps)

    return {
        "TP": float(TP), "FP": float(FP), "FN": float(FN), "TN": float(TN),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "iou": float(iou),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="", help="CrackForest root folder (optional if img_dir/gt_dir given)")
    ap.add_argument("--img_dir", type=str, default="", help="Images folder")
    ap.add_argument("--gt_dir", type=str, default="", help="GT masks folder")
    ap.add_argument("--p_low", type=float, default=97.0)
    ap.add_argument("--p_high", type=float, default=99.5)
    ap.add_argument("--save_pred", type=str, default="", help="Optional folder to save predicted masks")
    ap.add_argument("--limit", type=int, default=0, help="Evaluate only first N images (0=all)")
    ap.add_argument("--verbose_every", type=int, default=50)
    args = ap.parse_args()

    if not args.img_dir or not args.gt_dir:
        if not args.base:
            raise SystemExit("Please provide --base or both --img_dir and --gt_dir.")
        base = Path(args.base)
        img_dir, gt_dir = find_default_dirs(base)
    else:
        img_dir, gt_dir = Path(args.img_dir), Path(args.gt_dir)

    out_dir = Path(args.save_pred) if args.save_pred else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    img_files = list_images(img_dir)
    if args.limit > 0:
        img_files = img_files[: args.limit]

    TP = FP = FN = TN = 0.0
    per_image: List[Dict[str, float]] = []

    for i, img_path in enumerate(img_files, 1):
        stem = img_path.stem

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Cannot read image: {img_path}")
            continue

        gt = load_gt_mask(gt_dir, stem)

        pred = predict_final_mask_v4(img, p_low=args.p_low, p_high=args.p_high)

        m = compute_metrics(pred, gt)
        TP += m["TP"]; FP += m["FP"]; FN += m["FN"]; TN += m["TN"]
        per_image.append(m)

        if out_dir:
            cv2.imwrite(str(out_dir / f"{stem}.png"), pred)

        if (i % args.verbose_every) == 0:
            eps = 1e-8
            precision = TP / (TP + FP + eps)
            recall = TP / (TP + FN + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
            iou = TP / (TP + FP + FN + eps)
            print(f"[{i}/{len(img_files)}] micro P={precision:.4f} R={recall:.4f} F1={f1:.4f} Acc={accuracy:.4f} IoU={iou:.4f}")

    eps = 1e-8
    micro_precision = TP / (TP + FP + eps)
    micro_recall = TP / (TP + FN + eps)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + eps)
    micro_acc = (TP + TN) / (TP + TN + FP + FN + eps)
    micro_iou = TP / (TP + FP + FN + eps)

    if per_image:
        macro_precision = float(np.mean([x["precision"] for x in per_image]))
        macro_recall = float(np.mean([x["recall"] for x in per_image]))
        macro_f1 = float(np.mean([x["f1"] for x in per_image]))
        macro_acc = float(np.mean([x["accuracy"] for x in per_image]))
        macro_iou = float(np.mean([x["iou"] for x in per_image]))
    else:
        macro_precision = macro_recall = macro_f1 = macro_acc = macro_iou = 0.0

    print("\n=== FINAL RESULTS ===")
    print(f"phasepack used: {USE_PHASEPACK}")
    print(f"Images evaluated: {len(per_image)}/{len(img_files)}")
    print(f"p_low={args.p_low}  p_high={args.p_high}")

    print("\n[Micro / global (sum TP/FP/FN/TN)]")
    print(f"Precision: {micro_precision:.6f}")
    print(f"Recall   : {micro_recall:.6f}")
    print(f"F1       : {micro_f1:.6f}")
    print(f"Accuracy : {micro_acc:.6f}")
    print(f"IoU      : {micro_iou:.6f}")

    print("\n[Macro / mean per-image]")
    print(f"Precision: {macro_precision:.6f}")
    print(f"Recall   : {macro_recall:.6f}")
    print(f"F1       : {macro_f1:.6f}")
    print(f"Accuracy : {macro_acc:.6f}")
    print(f"IoU      : {macro_iou:.6f}")


if __name__ == "__main__":
    main()
