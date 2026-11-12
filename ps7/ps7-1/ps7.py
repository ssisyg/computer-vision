import sys
import cv2
import numpy as np

F, BASELINE, WIDTH = 777.0, 100.0, 665

def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def align(left, right):
    h, w = left.shape[:2]
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(gray(left), None)
    k2, d2 = sift.detectAndCompute(gray(right), None)
    good = [m for m, n in cv2.BFMatcher().knnMatch(d1, d2, k=2) if m.distance < 0.7 * n.distance]
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    _, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 1.0, 0.99)
    p1, p2 = p1[mask.ravel() == 1], p2[mask.ravel() == 1]
    if np.median(np.abs(p1[:, 1] - p2[:, 1])) > 1.5:
        a, b, c = np.linalg.lstsq(np.c_[p2, np.ones(len(p2))], p1[:, 1], rcond=None)[0]
        right = cv2.warpAffine(right, np.float32([[1, 0, 0], [a, b, c]]), (w, h))
    return left, right, p1[:, 0] - p2[:, 0]

def disparity(left, right):
    left, right, dx = align(left, right)
    low = int(np.floor(np.percentile(dx, 1))) - 8
    count = int(np.ceil((np.percentile(dx, 99) + 8 - low) / 16) * 16)
    sgbm = cv2.StereoSGBM_create(low, count, 5, 8 * 3 * 25, 32 * 3 * 25, 1, 0, 10, 100, 2,
                                 cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    d = sgbm.compute(gray(left), gray(right)).astype(np.float32) / 16
    d[d < max(low, 1)] = 0
    return left, d

def save_ply(name, left, d):
    h, w = d.shape
    ys, xs = np.nonzero(d)
    z = d[ys, xs]
    pts = np.c_[(xs - w / 2) * BASELINE / z, -(ys - h / 2) * BASELINE / z, -F * BASELINE / z]
    rgb = left[ys, xs][:, ::-1]
    header = ('ply\nformat ascii 1.0\ncomment made for CVE\nelement vertex %d\n'
              'property float32 x\nproperty float32 y\nproperty float32 z\n'
              'property uint8 red\nproperty uint8 green\nproperty uint8 blue\nend_header\n' % len(pts))
    with open(name + '.ply', 'w') as f:
        f.write(header)
        np.savetxt(f, np.c_[pts, rgb], fmt='%.3f %.3f %.3f %d %d %d')
    return len(pts)

if __name__ == '__main__':
    for name in sys.argv[1:]:
        left, right = cv2.imread(name + '-left.png'), cv2.imread(name + '-right.png')
        if left.shape[1] > 700:
            size = (WIDTH, round(left.shape[0] * WIDTH / left.shape[1]))
            left, right = cv2.resize(left, size, interpolation=cv2.INTER_AREA), cv2.resize(right, size, interpolation=cv2.INTER_AREA)
        left, d = disparity(left, right)
        valid = d[d > 0]
        lo, hi = np.percentile(valid, [2, 98])
        cv2.imwrite(name + '-disparity.png', np.where(d > 0, np.clip((d - lo) / (hi - lo), 0, 1) * 235 + 20, 0).astype(np.uint8))
        print(f'{name}: {save_ply(name, left, d)} points, disparity {valid.min():.1f}..{valid.max():.1f}')
