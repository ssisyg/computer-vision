import cv2
import numpy as np
from scipy.signal import convolve2d

thresholds = {
    "cheerios.png":  50,
    "professor.png": 40,
    "gear.png":      45,
    "circuit.png":   30,
}

Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
Gy = Gx.T

for name, th in thresholds.items():
    img = cv2.imread(name, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    gx = convolve2d(img, Gx, mode="same", boundary="symm")
    gy = convolve2d(img, Gy, mode="same", boundary="symm")
    mag = np.sqrt(gx**2 + gy**2)
    mag = (mag / mag.max() * 255).astype(np.uint8)
    edges = np.where(mag > th, 0, 255).astype(np.uint8)
    out = name.replace(".png", "-sobel.png")
    cv2.imwrite(out, edges)
    print(out, "saved")