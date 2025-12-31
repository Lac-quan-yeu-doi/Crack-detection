# Concrete Crack Detection using Classical Image Processing

A simple crack detection project on concrete surfaces using only image processing technique.

This repository compares several classic operation (Sobel, Morphological operations, Bottom-hat, Canny, Histogram Clipping) to detect cracks on concrete surface.

Moreover, it also conducts some furthur experiments on inpainting crack regions from the binary masks and blending few crack images images together to create a panorama.

## Structure

- `main` branch contains most of the experiments.
- `danh` branch contain experiments about phase congruency approach (`phase_congruency.ipynb`) in crack detection, and slight experiment about using YOLO to detect cracks.

## Dataset

- `dataset`: experiment dataset - CrackForest (source: [CrackForest Dataset](https://github.com/cuilimeng/CrackForest-dataset))
- `example`: some testing samples
- `real_life_image`: some collected real life images

## Files

- Experiment files: `*.ipynb`
  - `crack_detection_exp.ipynb`: experiments on crack detections
  - `phase_congruency.ipynb`: experiments on crack detections with phase congruency (in branch `danh`)
  - `inpaint_exp.ipynb`: experiments on crack inpainting
- `blending_experiments.py`: experiments on blending images into panorama and analyze it
- `utils.py`: contains organized code collected from the experiment files
- `main.py`: main code to call functions and method from `utils.py`

## Requirements

- OpenCV, NumPy, Matplotlib, scikit-image, SciPy

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Lac-quan-yeu-doi/Crack-detection.git
cd Crack-detection

# Install dependencies
pip install -r requirements.txt
```
