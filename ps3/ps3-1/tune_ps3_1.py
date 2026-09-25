import argparse
import cv2
import numpy as np

par = argparse.ArgumentParser()
par.add_argument("-i", "--input", required=True)
arg = par.parse_args()
img = cv2.imread(arg.input)

cv2.namedWindow("Tuner", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tuner", 1200, 700)
cv2.createTrackbar("Type", "Tuner", 0, 3, lambda x: None)
cv2.createTrackbar("K", "Tuner", 1, 15, lambda x: None)
cv2.createTrackbar("Sharp", "Tuner", 0, 30, lambda x: None)

while True:
    t = cv2.getTrackbarPos("Type", "Tuner")
    k = cv2.getTrackbarPos("K", "Tuner") | 1
    amount = cv2.getTrackbarPos("Sharp", "Tuner") / 10.0
    smoothed = [img, cv2.GaussianBlur(img, (k, k), 0), cv2.medianBlur(img, k), cv2.bilateralFilter(img, k, 75, 75)][t]
    blur = cv2.GaussianBlur(smoothed, (0, 0), 3)
    result = cv2.addWeighted(smoothed, 1 + amount, blur, -amount, 0)
    cv2.imshow("Tuner", np.hstack([img, result]))
    if cv2.waitKey(50) == ord('q'):
        break

print(f"type={t}, k={k}, sharpen={amount}")
cv2.imwrite("test_output.png", result)