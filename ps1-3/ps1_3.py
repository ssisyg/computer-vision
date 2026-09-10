import cv2
import os
import numpy as np

f = input("Enter image name:").strip()
ig = cv2.imread(f)
b, ext = os.path.splitext(f)
cv2.imshow("Original", ig); cv2.waitKey(1)

corrected = ig.copy()
while True:
    g = input("Enter gama value").strip()
    if g == "done":
        break

    gamma = float(g)
    corrected = (np.power(ig.astype(np.float32) / 255.0, gamma) * 255).astype(np.uint8)
    cv2.imshow("Gamma corrected", corrected); cv2.waitKey(1)

cv2.imwrite(f"{b}_gcorrected{ext}", corrected)

cv2.waitKey(0)
cv2.destroyAllWindows()