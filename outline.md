# Classical vs. Deep Learning Approaches for Concrete Crack Detection

**A Comparative Study Using Sobel Edge Detection + Morphology and Yolov7**  
[Your Name] – [Student ID] – [Course Name] – [Date]

## 1. Introduction

- What is Crack detection? (brief)
- Importance of automated crack detection in civil infrastructure (brief)
- Limitations of manual inspection (time, cost, human error) (brief)
- Two main paradigms: Classical Computer Vision (zero training) vs. Deep Learning (emphasize)
- Objective of the project: Choose app Compare a simple, interpretable classical method (Sobel + Morphology) with the state-of-the-art Yolov7 (emphasize)

## 2. Literature Survey

### 2.1 Classical (Non-Deep Learning) Methods (brief)

- Gradient-based edge detectors (Sobel, Prewitt, Roberts, Canny)
- Second-order methods (Laplacian of Gaussian, Difference of Gaussians)
- Morphological approaches (Top-Hat, Bottom-Hat, reconstruction)
- Advanced classical methods (Frangi, Hessian + Minimal Path, Phase Congruency)

### 2.2 Deep Learning Methods (brief)

- Evolution from CNN → FCN → U-Net → DeepCrack
- Object-detection-based crack detection using YOLO series
- Advantages of data-driven approaches

### 2.3 Motivation for Comparing Sobel with Yolov7 (emphasize)

- Sobel = most fundamental, fully interpretable, zero-training method
- Yolov7 = current real-time SOTA

### 2.4 The reason for picking your approach (emphasize)

- Investigate the pros and cons of non-deep learning method
- Comparison between classic approach and deep learning approach

## 3. Methodology (emphasize)

### 3.1 Proposed Classical Approach (Main Method) – Sobel + Morphology

- Pipeline overview (block diagram)
- CLAHE contrast enhancement
- Sobel gradient computation (mathematical kernels)
- Gradient magnitude & Otsu thresholding
- Morphological closing + small object removal
- Detailed explanation of each step

### 3.2 Deep Learning Baseline – Yolov7 (brief)

- Overview Yolo
- Dataset used (e.g., CFD, SDNET2018, custom)
- Training configuration (epochs, batch size, augmentation)
- Inference pipeline

## 4. Experiments and Results

### 4.1 Experiments

- Dataset description
- Qualitative results (side-by-side images: Original → Sobel → Canny → Yolov7)
- Quantitative comparison table (speed, accuracy, memory)
- Failure cases analysis for both methods

### 4.2 Results

- Visual comparison
- Precision, Recall, F1-score, IOU, Accuracy của mask
- Inference speed (ms/image) and model size

## 5. Discussion & Personal Opinion

- Strengths and weaknesses of Sobel + Morphology
- Strengths and weaknesses of Yolov7
- When to use classical vs. deep learning methods
- Proposed improvements & future ideas:
  1. Multi-scale Sobel
  2. Hybrid pipeline (Sobel pseudo-labels → lightweight CNN)
  3. Two-stage detection (Sobel screening → YOLO refinement)
  4. Adaptive percentile thresholding + region growing

## 6. Conclusion

- Summary of findings
  - Classical methods remain highly relevant in resource-constrained scenarios
  - Best approach depends on application requirements

## 7. References

## 8. Appendix – Code & Implementation

- Include code link
- Full classical implementation (Sobel + Canny)
- Yolov7 training/inference script (or notebook link)
- Declaration: “This project was implemented individually by [Your Name]”
