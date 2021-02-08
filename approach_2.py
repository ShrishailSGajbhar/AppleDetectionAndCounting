import cv2 
import numpy as np 
import argparse 

# Defining the color ranges to be filtered.
# The following ranges should be used on HSV domain image.
low_apple_red = (160.0, 153.0, 153.0)
high_apple_red = (180.0, 255.0, 255.0)
low_apple_raw = (0.0, 150.0, 150.0)
high_apple_raw = (15.0, 255.0, 255.0)

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required=True, help="Path to input image")
args = vars(ap.parse_args())

image_bgr = cv2.imread(args['image'])
image = image_bgr.copy()
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

mask_red = cv2.inRange(image_hsv,low_apple_red, high_apple_red)
mask_raw = cv2.inRange(image_hsv,low_apple_raw, high_apple_raw)

mask = mask_red + mask_raw


cnts,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
c_num=0
for i,c in enumerate(cnts):
    # draw a circle enclosing the object
    ((x, y), r) = cv2.minEnclosingCircle(c)
    if r>34:
        c_num+=1
        cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(image, "#{}".format(c_num), (int(x) - 10, int(y)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    else:
        continue

cv2.imshow("Original image", image_bgr)
cv2.imshow("Detected Apples", image)
cv2.imshow("HSV Image", image_hsv)
cv2.imshow("Mask image", mask)
cv2.waitKey(0)
cv2.imwrite("image6.png", image_bgr)
cv2.imwrite("pic6.png", image)
cv2.imwrite("hsv.png", image_hsv)
cv2.imwrite("mask.png", mask)
