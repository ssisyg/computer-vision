import cv2
import numpy as np

AREA_MIN, PAD, WIDTH = 150, 4, 1800

def find_blobs(binary):
    cnts, hier = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hier = hier[0]
    blobs = []
    for i, c in enumerate(cnts):
        if hier[i][3] != -1 or cv2.contourArea(c) < AREA_MIN:
            continue
        x, y, w, h = cv2.boundingRect(c)
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1, offset=(-x, -y))
        for j in range(len(cnts)):
            if hier[j][3] == i:
                cv2.drawContours(mask, [cnts[j]], -1, 0, -1, offset=(-x, -y))
        blobs.append((cv2.countNonZero(mask), x, y, mask))
    return sorted(blobs, key=lambda b: b[0], reverse=True)

def horizontal(mask):
    m = cv2.moments(mask, binaryImage=True)
    theta = 0.5 * np.arctan2(2 * m['mu11'], m['mu20'] - m['mu02'])
    h, w = mask.shape
    s = int(np.ceil(np.hypot(w, h))) + 4
    canvas = np.zeros((s, s), np.uint8)
    canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = mask
    R = cv2.getRotationMatrix2D((s / 2, s / 2), np.degrees(theta), 1.0)
    rot = cv2.warpAffine(canvas, R, (s, s), flags=cv2.INTER_LINEAR) > 127
    ys, xs = np.nonzero(rot)
    return rot[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

def catalog(tiles):
    rows, row, x = [], [], 0
    for t in tiles:
        if row and x + t.shape[1] + 2 * PAD > WIDTH:
            rows.append(row)
            row, x = [], 0
        row.append(t)
        x += t.shape[1] + 2 * PAD
    rows.append(row)
    heights = [max(t.shape[0] for t in r) + 2 * PAD for r in rows]
    img = np.full((sum(heights) + 1, WIDTH, 3), 255, np.uint8)
    y = 0
    for r, rh in zip(rows, heights):
        x = 0
        for t in r:
            cw = t.shape[1] + 2 * PAD
            ty = y + (rh - t.shape[0]) // 2
            img[ty:ty + t.shape[0], x + PAD:x + PAD + t.shape[1]][t] = 0
            cv2.rectangle(img, (x, y), (x + cw, y + rh), (160, 160, 160), 1)
            x += cw
        y += rh
    return img

for name, ext in [('blood-cells1', 'jpg'), ('blood-cells2', 'png'), ('blood-cells3', 'png')]:
    img = cv2.imread(f'{name}.{ext}')
    _, binary = cv2.threshold(cv2.GaussianBlur(img[:, :, 1], (5, 5), 0), 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    cv2.imwrite(f'{name}-binary.jpg', 255 - binary)

    blobs = find_blobs(binary)
    rng = np.random.default_rng(0)
    vis = np.full(img.shape, 255, np.uint8)
    for _, x, y, mask in blobs:
        h, w = mask.shape
        vis[y:y + h, x:x + w][mask > 0] = rng.integers(0, 200, 3)
    cv2.imwrite(f'{name}-blobs.jpg', vis)

    cat = catalog([horizontal(b[3]) for b in blobs])
    cv2.imwrite(f'{name}-catalog.jpg', cat, [cv2.IMWRITE_JPEG_QUALITY, 95])
    areas = [b[0] for b in blobs]
    print(f'{name}: {len(blobs)} blobs, area max {max(areas)}, min {min(areas)}, catalog {cat.shape[1]}x{cat.shape[0]}')
