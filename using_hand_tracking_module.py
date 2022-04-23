import cv2
import numpy as np
import mediapipe as mp
import time
import Hand_Tracking_Module as htm
import math


pTime = 0
cTime = 0
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = htm.HandDetector()

while True:
    success, img = cap.read()
    img = cv2.flip(img,1 )
    img = detector.findHands(img)
    lmlist = detector.findPosition( img , draw = True) #here draw = False removing custom landmarks(circles)


    # if len(lmlist) != 0:
    #     print(lmlist[4])

    if len(lmlist) != 0:
        ang = detector.findAngle(img , 8 , 12 , 5 ,9 , draw = True )
        print(ang)

    ######### To find FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (18, 70), cv2.FONT_HERSHEY_SIMPLEX,
                3, (255, 0, 255), 3)
    #########

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()