import cv2
import numpy as np
from ps7 import disparity

left, right = cv2.imread('ball-left.png'), cv2.imread('ball-right.png')
left, d = disparity(left, right)
h, w = d.shape

green = cv2.inRange(cv2.cvtColor(left, cv2.COLOR_BGR2HSV), (35, 30, 40), (90, 255, 255))
green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
(_, _), radius = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))

ys, xs = np.nonzero((cv2.erode(green, np.ones((41, 41), np.uint8)) > 0) & (d > 0))
u, v, disp = xs - w / 2, ys - h / 2, d[ys, xs].astype(np.float64)

def sphere_error(f):
    p = np.c_[u / disp, v / disp, f / disp]
    sol = np.linalg.lstsq(np.c_[2 * p, np.ones(len(p))], (p ** 2).sum(axis=1), rcond=None)[0]
    center = sol[:3]
    r = np.sqrt(sol[3] + (center ** 2).sum())
    return r / np.linalg.norm(center) - radius / np.hypot(radius, f)

fs = np.arange(300, 1500)
err = np.array([sphere_error(f) for f in fs])
print('F =', fs[np.argmin(np.abs(err))])
