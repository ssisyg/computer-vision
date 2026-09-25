import argparse
import cv2
import numpy as np

par = argparse.ArgumentParser()
par.add_argument("-i", "--input", required=True)
arg = par.parse_args()
img = cv2.imread(arg.input, cv2.IMREAD_GRAYSCALE)
img = cv2.GaussianBlur(img, (5, 5), 0)

cv2.namedWindow("Tuner", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tuner", 1200, 700)
cv2.createTrackbar("Th1", "Tuner", 50, 500, lambda x: None)
cv2.createTrackbar("Th2", "Tuner", 150, 500, lambda x: None)
cv2.createTrackbar("Aperture", "Tuner", 1, 2, lambda x: None)   # 0->3, 1->5, 2->7
cv2.createTrackbar("L2grad", "Tuner", 0, 1, lambda x: None)

while True:
    t1 = cv2.getTrackbarPos("Th1", "Tuner")
    t2 = cv2.getTrackbarPos("Th2", "Tuner")
    aperture = [3, 5, 7][cv2.getTrackbarPos("Aperture", "Tuner")]
    l2 = bool(cv2.getTrackbarPos("L2grad", "Tuner"))

    edges = cv2.Canny(img, t1, t2, apertureSize=aperture, L2gradient=l2)
    result = 255 - edges  # black edges on white background

    cv2.imshow("Tuner", np.hstack([img, result]))
    if cv2.waitKey(50) == ord('q'):
        break

print(f"th1={t1}, th2={t2}, aperture={aperture}, l2={l2}")
cv2.imwrite("canny_test.png", result)