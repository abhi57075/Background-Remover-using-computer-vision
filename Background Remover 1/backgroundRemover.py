# Media pipe package -> selfie segmentation

import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS, 60) # increasing the frame rate to 60 fps

segmentor = SelfiSegmentation(1) # 0 will be for generalized (a bit slow) ; 1 will be for landscapes (a bit fast)
fpsReader = cvzone.FPS()

imgBg = cv2.imread("backgrounds/room1.jpg")
imgBg = cv2.resize(imgBg, (640,480))

listImg = os.listdir("backgrounds")
imgList = []

for imgPath in listImg:
    img = cv2.imread(f"backgrounds/{imgPath}")
    img = cv2.resize(img, (640, 480))
    imgList.append(img)

print(len(imgList))

indexImg = 0

while 1:
    success, img = cap.read()
   
    imgOut1 = segmentor.removeBG(img, (255,0,0), threshold = 0.7) # Putting color as a background, not a compulsion to write threshold but we can do so to sharpen the outline of ourselves, the value should be between 0 and 1
    imgOut2 = segmentor.removeBG(img, imgBg, threshold = 0.7)
    imgOut3 = segmentor.removeBG(img, imgList[indexImg], threshold = 0.2)


    imgStacked1 = cvzone.stackImages([img, imgOut1, imgOut2],3,0.8) # 2 columns and the scale is 1
    fps1, imgStacked1 = fpsReader.update(imgStacked1, color = (0,0,255))
    
    cv2.imshow("Image",imgStacked1)

    imgStacked2 = cvzone.stackImages([img, imgOut3],2,1) # 2 columns and the scale is 1
    fps1, imgStacked2 = fpsReader.update(imgStacked2, color = (0,0,255))

    cv2.imshow("Dynamic Image",imgStacked2)
    
    key = cv2.waitKey(1)
    if key == ord('p'):
        if indexImg > 0:
            indexImg-=1
        else:
            indexImg = len(imgList)-1
    elif key == ord('n'):
        if indexImg < len(imgList)-1:
            indexImg+=1
        else:
            indexImg = 0
    elif key == ord('q'):
        break