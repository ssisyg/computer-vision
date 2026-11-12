import sys
import cv2
import numpy as np

SIZE = 600
NEAR = 0.6

def load(name):
    data = np.loadtxt(name + '.ply', skiprows=12)
    data = data[data[:, 2] >= np.quantile(data[:, 2], 1 - NEAR)]
    return data[:, :3], data[:, 3:6].astype(np.uint8)[:, ::-1]

def render(pts, colors, yaw, pitch=0):
    p = pts - np.median(pts, axis=0)
    a, b = np.radians(yaw), np.radians(pitch)
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    p = p @ (rx @ ry).T
    lo, hi = np.percentile(p[:, :2], 1, axis=0), np.percentile(p[:, :2], 99, axis=0)
    scale = 0.9 * SIZE / max(hi - lo)
    px = ((p[:, 0] - (lo[0] + hi[0]) / 2) * scale + SIZE / 2).astype(int)
    py = (SIZE / 2 - (p[:, 1] - (lo[1] + hi[1]) / 2) * scale).astype(int)
    order = np.argsort(p[:, 2])
    img = np.full((SIZE, SIZE, 3), 255, np.uint8)
    for dx in (0, 1):
        for dy in (0, 1):
            x, y = px[order] + dx, py[order] + dy
            ok = (x >= 0) & (x < SIZE) & (y >= 0) & (y < SIZE)
            img[y[ok], x[ok]] = colors[order][ok]
    return img

if __name__ == '__main__':
    for name in sys.argv[1:]:
        pts, colors = load(name)
        views = [render(pts, colors, yaw) for yaw in (0, 35, -35, 70)]
        cv2.imwrite(name + '-views.png', np.vstack([np.hstack(views[:2]), np.hstack(views[2:])]))
