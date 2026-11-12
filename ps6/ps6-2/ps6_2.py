import cv2
import numpy as np

THRESHOLD, DEFECT = 110, 0.04

img = cv2.imread('spade-terminal.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)
binary = cv2.dilate(cv2.erode(binary, None), None)
cont, hier = cv2.findContours(255 - binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
parts = [c for c, h in zip(cont, hier[0]) if h[3] == -1 and 3000 < cv2.contourArea(c) < 20000]

n = len(parts)
dist = np.array([[cv2.matchShapes(a, b, cv2.CONTOURS_MATCH_I3, 0) for b in parts] for a in parts])
template = int(np.argmin(dist.sum(axis=1)))
defective = [i for i in range(n) if dist[template, i] > DEFECT]

out = img.copy()
cv2.drawContours(out, [parts[i] for i in defective], -1, (0, 0, 255), -1)
cv2.imwrite('spade-terminal-output.png', out)
print(f'{n} terminals, template #{template}, {len(defective)} defective')
print('distances to template:', np.round(np.sort(dist[template])[::-1][:6], 4))
