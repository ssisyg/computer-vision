# 24-678: Computer Vision for Engineers

Carnegie Mellon University — Coursework and problem sets for 24-678 Computer Vision for Engineers.

## Environment

- OS: Windows 11 with WSL2 (Ubuntu 26.04.1 LTS)
- IDE: Visual Studio Code (Remote-WSL)
- Python environment: conda env `cve` (Python 3.12, opencv-python, numpy, scikit-learn, scikit-image, scipy, open3d)

## Problem Sets

| PS  | Topic                                                  | Due Date | Status      |
|-----|--------------------------------------------------------|----------|-------------|
| PS1 | Thresholding, color conversion, gamma correction        | 9/11     | Done        |
| PS2 | Pseudocoloring (LUT-based color mapping)                | 9/18     | Done        |
| PS3 | Image improvement, Sobel & Canny edge detection         | 9/25     | Done        |
| PS4 | Image stitching                                         | 10/2     | Done        |
| PS5 | Blob detection and contour analysis                     | 10/23    | Done        |
| PS6 | Mechanical part recognition                             | 11/6     | Done        |
| PS7 | Stereo vision (disparity and 3D reconstruction)         | 11/13    | Done        |
| PS8 | TBD                                                     | 12/12    | Not started |

## Repository Structure

Each problem set has its own folder (`psN/`) containing the submitted
sub-folders (`psN-k/`) with the source code, input/output images, and a
`readme.txt` (OS, IDE, hours spent). Large files such as point clouds
(`*.ply`) and zip archives are not tracked.

```
ps1/ps1-2, ps1-3
ps2/ps2-1
ps3/ps3-1, ps3-2
ps4/ps4-1
ps5/ps5-1, ps5-2
ps6/ps6-1, ps6-2
ps7/ps7-1
```

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

## PS4 Summary

- **ps4-1**: Stitches image sets (`door`, `house`, `pittsburgh`, `wall`) into
  stitched panoramas (`*-stitched.png`) with `stitch.py`, and saves the
  per-set results to `*-result.json`.

## PS5 Summary

- **ps5-1**: Analyzes two wall images (`wall1`, `wall2`) with blob detection
  and contour extraction and marks the detected cracks
  (`*-blobs.png`, `*-contours.png`, `*-cracks.png`), using `ps5_1.py`.
- **ps5-2**: Processes three blood-cell images: binarization, blob detection,
  and a catalog image of the detected cells for each (`*-binary.jpg`,
  `*-blobs.jpg`, `*-catalog.jpg`), using `ps5_2.py`.

## PS6 Summary

- **ps6-1**: Recognizes the mechanical parts in the provided images and
  produces an output image with all parts labeled (`all-parts-output.png`),
  using `ps6_1.py`.
- **ps6-2**: Analysis of the spade terminal part, with the terminal output
  saved as `spade-terminal-output.png`, using `ps6_2.py`.

## PS7 Summary

- **ps7-1**: Stereo vision on parallel stereo pairs (`baby`, `ball`, `plant`,
  and a self-taken phone pair `junlongn`). The disparity is computed with
  OpenCV `StereoSGBM` (search range set from SIFT matches) and saved as a
  grayscale image on the left camera frame (closer is lighter). The disparity
  is reprojected to a colored ASCII PLY point cloud with $Z = fB/d$. With no
  camera parameters given, the focal length ($f = 777$ pixels) is found from
  the bowling-ball pair by making the reconstructed ball match a sphere, and
  is reused for the other pairs. For the phone pair, a vertical-only affine
  alignment corrects the hand-held misalignment. `calibrate.py` estimates
  $f$, and `render.py` renders views of the PLY files.