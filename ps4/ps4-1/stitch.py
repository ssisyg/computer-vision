import os
import cv2
import numpy as np
import json
import argparse

def savePick():
    global pick, json_name
    with open(json_name, 'w') as outfile:
        json.dump({"pick": pick}, outfile)

def combine():
    global result, imageC, imageL, imageR, pick, out_name
    (h, w) = imageC.shape[:2]

    cng = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    th, mask_c = cv2.threshold(cng, 1, 255, cv2.THRESH_BINARY)
    mask_c = mask_c / 255

    # right: warp imageR into center's coordinate frame using picked points
    src_pnts = np.empty([4, 2], np.float32)
    dst_pnts = np.empty([4, 2], np.float32)
    for i in range(4):
        src_pnts[i][0] = float(pick[0][i][0])
        src_pnts[i][1] = float(pick[0][i][1])
        dst_pnts[i][0] = float(pick[1][i][0] + w)
        dst_pnts[i][1] = float(pick[1][i][1] + h)
    M_r = cv2.getPerspectiveTransform(src_pnts, dst_pnts)
    rn = cv2.warpPerspective(imageR, M_r, (w * 3, h * 3))
    rng = cv2.cvtColor(rn, cv2.COLOR_BGR2GRAY)
    th, mask_r = cv2.threshold(rng, 1, 255, cv2.THRESH_BINARY)
    mask_r = mask_r / 255

    # left: warp imageL into center's coordinate frame using picked points
    src_pnts_l = np.empty([4, 2], np.float32)
    dst_pnts_l = np.empty([4, 2], np.float32)
    for i in range(4):
        src_pnts_l[i][0] = float(pick[2][i][0])
        src_pnts_l[i][1] = float(pick[2][i][1])
        dst_pnts_l[i][0] = float(pick[3][i][0] + w)
        dst_pnts_l[i][1] = float(pick[3][i][1] + h)
    M_l = cv2.getPerspectiveTransform(src_pnts_l, dst_pnts_l)
    ln = cv2.warpPerspective(imageL, M_l, (w * 3, h * 3))
    lng = cv2.cvtColor(ln, cv2.COLOR_BGR2GRAY)
    th, mask_l = cv2.threshold(lng, 1, 255, cv2.THRESH_BINARY)
    mask_l = mask_l / 255

    # alpha blending: weight = 1 / (number of overlapping images at that pixel)
    mask = np.array(mask_c + mask_l + mask_r, float)
    ag = 1.0 / np.maximum(1, mask)

    result[:, :, 0] = result[:, :, 0] * ag + ln[:, :, 0] * ag + rn[:, :, 0] * ag
    result[:, :, 1] = result[:, :, 1] * ag + ln[:, :, 1] * ag + rn[:, :, 1] * ag
    result[:, :, 2] = result[:, :, 2] * ag + ln[:, :, 2] * ag + rn[:, :, 2] * ag

    cv2.imwrite(out_name, result)
    cv2.imshow("result", result)
    print(f"Saved {out_name}")

def right_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        mousePick(x, y, 0)

def center_click_r(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        mousePick(x, y, 1)

def left_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        mousePick(x, y, 2)

def center_click_l(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        mousePick(x, y, 3)

def mousePick(x, y, idx):
    global rn, cn, ln, imageR, imageC, imageL, pick
    if idx == 0:
        src, wn = imageR, "right"
    elif idx == 1 or idx == 3:
        src, wn = imageC, "center"
    elif idx == 2:
        src, wn = imageL, "left"

    pick[idx].append((x, y))
    dst = src.copy()
    col = (0, 0, 255) if idx < 2 else (255, 0, 0)  # red = right side, blue = left side
    for i in range(len(pick[idx])):
        dst = cv2.circle(dst, pick[idx][i], 5, col, 2)
        dst = cv2.putText(dst, str(i), (pick[idx][i][0] + 10, pick[idx][i][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, col, 1)
    cv2.imshow(wn, dst)
    cv2.waitKey(1)

    if len(pick[idx]) >= 4:
        print('Is it OK? (y/n)')
        ans = input()
        if ans.lower() == 'y':
            if idx >= 3:
                savePick()
                combine()
            elif idx == 0:
                print('center 4 points (matching right)')
                cv2.setMouseCallback("center", center_click_r)
            elif idx == 1:
                print('left 4 points')
                cv2.setMouseCallback("left", left_click)
            elif idx == 2:
                print('center 4 points (matching left)')
                cv2.setMouseCallback("center", center_click_l)
        else:
            pick[idx] = []
            cv2.imshow(wn, src.copy())

parser = argparse.ArgumentParser(description='Combine 3 images')
parser.add_argument('-d', '--data', type=int, help='0=wall,1=door,2=house,3=pittsburgh', default=3)
args = parser.parse_args()
dataset = args.data

datasets = {
    0: "wall", 1: "door", 2: "house", 3: "pittsburgh"
}
name = datasets[dataset]
ext = "png" if name == "wall" else "jpg"
imageL = cv2.imread(f"{name}-left.{ext}")
imageC = cv2.imread(f"{name}-center.{ext}")
imageR = cv2.imread(f"{name}-right.{ext}")

out_name = f"{name}-stitched.png"
json_name = f"{name}-result.json"

result = cv2.copyMakeBorder(imageC, imageC.shape[0], imageC.shape[0], imageC.shape[1], imageC.shape[1],
                             borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
print(imageL.shape, imageC.shape, imageR.shape, result.shape)

cv2.namedWindow("left", cv2.WINDOW_NORMAL)
cv2.namedWindow("center", cv2.WINDOW_NORMAL)
cv2.namedWindow("right", cv2.WINDOW_NORMAL)
cv2.namedWindow("result", cv2.WINDOW_NORMAL)

ln, cn, rn = imageL.copy(), imageC.copy(), imageR.copy()
cv2.imshow("left", ln)
cv2.imshow("center", cn)
cv2.imshow("right", rn)
cv2.imshow("result", result)

pick = [[], [], [], []]

if os.path.exists(json_name):
    print('use saved points? (y/n)')
    if input().lower() == 'y':
        with open(json_name) as f:
            pick = json.load(f)["pick"]
        combine()
    else:
        print("right 4 points")
        cv2.setMouseCallback("right", right_click)
else:
    print("right 4 points")
    cv2.setMouseCallback("right", right_click)

cv2.waitKey()
cv2.destroyAllWindows()
