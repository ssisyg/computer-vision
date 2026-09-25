import argparse
import cv2
import numpy as np
import os

#step1
par = argparse.ArgumentParser()
par.add_argument("-i", "--input", required = True)
arg = par.parse_args()
gray = cv2.imread(arg.input, cv2.IMREAD_GRAYSCALE)


cv2.imshow("Input", gray)
cv2.waitKey(0)

#step2
g_min = np.amin(gray)
g_max = np.amax(gray)
max_p = list(zip(*np.where(gray == g_max)))

print(f"min = {g_min}, max = {g_max}")
print(f"max is {len(max_p)} pixel(s): {max_p}")

#step3
level = np.arange(256, dtype = np.float64)
t = np.clip((level - g_min) / (g_max - g_min), 0.0, 1.0)
blue = np.clip(255.0 * (1.0 - 2.0 * t), 0, 255)
green = np.clip(255.0 * (1.0 - np.abs(2.0 * t - 1.0)), 0, 255)
red = np.clip(255.0 * (2.0 * t - 1.0), 0, 255)
l = np.stack([blue, green, red], axis = 1).astype(np.uint8)
color = l[gray]

print(l[g_min], l[(g_min + g_max) // 2], l[g_max])

#step4
cy, cx = np.mean(max_p, axis = 0).astype(int)
center = (cx, cy)
white = (255, 255, 255)
rad = 12
cv2.circle(color, center, rad, white, 2)
cv2.line(color, (cx - rad, cy), (cx + rad, cy), white, 2)
cv2.line(color, (cx, cy - rad), (cx, cy + rad), white, 2)

#step5
base, _ = os.path.splitext(arg.input)
out = f"{base}-color.png"
cv2.imwrite(out, color)
cv2.imshow("output", color)
cv2.waitKey(0)
