import cv2
import numpy as np

AREA_MIN, LENGTH_MIN = 1000, 100

def thinning(mask):
    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    thin = np.zeros_like(mask)
    while cv2.countNonZero(mask):
        er = cv2.erode(mask, k)
        thin |= er - cv2.morphologyEx(er, cv2.MORPH_OPEN, k)
        mask = er
    return thin

for name in ['wall1', 'wall2']:
    gray = cv2.imread(name + '.png', 0)
    fg = np.where(gray < 128, 255, 0).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blobs = cv2.erode(cv2.dilate(fg, k, iterations=2), k, iterations=2)
    cv2.imwrite(name + '-blobs.png', 255 - blobs)

    cnts, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rng = np.random.default_rng(0)
    out = np.full((*gray.shape, 3), 255, np.uint8)
    for c in cnts:
        cv2.drawContours(out, [c], -1, [int(v) for v in rng.integers(0, 200, 3)], 2)
    cv2.imwrite(name + '-contours.png', out)

    keep = np.zeros_like(blobs)
    cracks = [c for c in cnts
              if cv2.contourArea(c) >= AREA_MIN and max(cv2.minAreaRect(c)[1]) >= LENGTH_MIN]
    cv2.drawContours(keep, cracks, -1, 255, -1)
    axis = thinning(keep)
    cv2.imwrite(name + '-cracks.png', 255 - axis)
    print(f'{name}: {len(cnts)} blobs, {len(cracks)} cracks, axis pixels {cv2.countNonZero(axis)}')
