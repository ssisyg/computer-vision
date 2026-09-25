# 24-678: Computer Vision for Engineers

Carnegie Mellon University — Coursework and problem sets for 24-678 Computer Vision for Engineers.

## Environment

- OS: Windows 11 with WSL2 (Ubuntu 26.04.1 LTS)
- IDE: Visual Studio Code (Remote-WSL)
- Python environment: conda env `cve` (Python 3.12, opencv-python, numpy, scikit-learn, scikit-image, scipy, open3d)

## Problem Sets

| PS  | Topic                                                 | Due Date | Status      |
|-----|--------------------------------------------------------|----------|-------------|
| PS1 | Thresholding, color conversion, gamma correction        | 9/11     | Done        |
| PS2 | Pseudocoloring (LUT-based color mapping)                | 9/18     | Done        |
| PS3 | Image improvement, Sobel & Canny edge detection         | 9/25     | Done        |
| PS4 | TBD                                                     | 10/2     | Not started |
| PS5 | TBD                                                     | 10/23    | Not started |
| PS6 | TBD                                                     | 11/6     | Not started |
| PS7 | TBD                                                     | 11/13    | Not started |
| PS8 | TBD                                                     | 12/12    | Not started |

## Repository Structure


## PS1 Summary

- **ps1-2**: Read a color image, convert to grayscale, apply binary thresholding
  (bright or dark region emphasis), and paint the target region red on the
  original color image.
- **ps1-3**: Apply gamma correction interactively to improve image exposure.

## PS2 Summary

- **ps2-1**: Hand-written LUT-based pseudocoloring — maps grayscale intensity
  to a blue-green-red color gradient without using OpenCV's built-in colormap
  functions, and marks the location of the maximum-intensity pixel(s) on the
  output image.

## PS3 Summary

- **ps3-1**: Improves four noisy/blurry images using OpenCV smoothing
  (Gaussian/median/bilateral blur) and unsharp-mask sharpening, tuned via an
  interactive trackbar GUI (`tune_ps3_1.py`).
- **ps3-2**: Edge detection on four images using (a) a hand-written Sobel
  filter (no `cv2.Sobel`) via `scipy.signal.convolve2d`, and (b) `cv2.Canny()`
  tuned through an interactive GUI (`tune_canny.py`) with sliders for
  threshold1, threshold2, aperture size, and L2 gradient.
