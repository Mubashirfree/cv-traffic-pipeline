import cv2
import sys

img_path = sys.argv[1] if len(sys.argv) > 1 else "data/calibration_ref_MVI_20011.jpg"
img = cv2.imread(img_path)
points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Click 4 points, then press any key", img)

cv2.imshow("Click 4 points, then press any key", img)
cv2.setMouseCallback("Click 4 points, then press any key", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("\nAll points:", points)
