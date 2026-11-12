import cv2
import numpy as np

RED, GREEN, BLUE = (0, 0, 255), (0, 255, 0), (255, 0, 0)
PURPLE, YELLOW = (128, 0, 128), (0, 255, 255)

def is_square(contour):
    w, h = cv2.minAreaRect(contour)[1]
    return abs(w - h) / max(w, h) < 0.1

def is_circle(contour):
    (_, _), r = cv2.minEnclosingCircle(contour)
    return abs(cv2.arcLength(contour, True) - 2 * np.pi * r) / (2 * np.pi * r) < 0.1

def classify(mask):
    cont, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    outer = max(cont, key=cv2.contourArea)
    holes = [c for c, h in zip(cont, hier[0]) if h[3] != -1 and cv2.contourArea(c) > 100]
    if not holes:
        return RED
    if is_circle(outer):
        return GREEN if is_circle(holes[0]) else PURPLE
    return YELLOW if is_square(outer) else BLUE

def split(mask, centers):
    ys, xs = np.nonzero(mask)
    d = [(xs - cx) ** 2 + (ys - cy) ** 2 for cx, cy in centers]
    nearest = np.argmin(d, axis=0)
    pieces = []
    for k in range(len(centers)):
        piece = np.zeros_like(mask)
        piece[ys[nearest == k], xs[nearest == k]] = 255
        pieces.append(piece)
    return pieces

img = cv2.imread('all-parts.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
binary = cv2.dilate(cv2.erode(binary, None), None)
cont, hier = cv2.findContours(255 - binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

out = img.copy()
counts = {}
for i, c in enumerate(cont):
    area = cv2.contourArea(c)
    if hier[0][i][3] != -1 or not 3000 < area < 100000:
        continue
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, cont, i, 255, -1)
    centers = []
    for j, h in enumerate(hier[0]):
        if h[3] == i:
            cv2.drawContours(mask, cont, j, 0, -1)
            m = cv2.moments(cont[j])
            if m['m00'] > 100:
                centers.append((m['m10'] / m['m00'], m['m01'] / m['m00']))
    for part in split(mask, centers) if area > 40000 else [mask]:
        color = classify(part)
        out[part > 0] = color
        counts[color] = counts.get(color, 0) + 1

cv2.imwrite('all-parts-output.png', out)
names = {RED: 'spade', GREEN: 'washer', BLUE: 'ring', PURPLE: 'internal-tooth', YELLOW: 'external-tooth'}
print({names[k]: v for k, v in counts.items()}, 'total', sum(counts.values()))
