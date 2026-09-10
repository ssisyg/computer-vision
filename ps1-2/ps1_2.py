    import cv2
    import os
    import numpy as np

    f = input("Enter image name:").strip()
    mode = input("bright or dark one?").strip()

    ig = cv2.imread(f)
    b, ext = os.path.splitext(f)
    cv2.imshow("input color", ig); cv2.waitKey(1)

    gray = cv2.cvtColor(ig, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale", gray); cv2.waitKey(1)
    cv2.imwrite(f"{b}_grayscale{ext}", gray)

    t = float(input("enter threshold value:"))
    _, binary = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY if mode == "bright" else cv2.THRESH_BINARY_INV)
    cv2.imshow("Binary", binary); cv2.waitKey(1)
    cv2.imwrite(f"{b}_binary{ext}", binary)

    output = ig.copy()
    output[binary == 255] = [0, 0, 255]
    cv2.imshow("Output", output); cv2.waitKey(1)
    cv2.imwrite(f"{b}_output{ext}", output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()