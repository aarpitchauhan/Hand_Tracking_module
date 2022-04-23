# import cv2
#
# # Read in image
# image = cv2.imread('lena.jpg')
#
# # Create ROI coordinates
# topLeft = (60, 140)
# bottomRight = (340, 250)
# x, y = topLeft[0], topLeft[1]
# w, h = bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1]
#
# # Grab ROI with Numpy slicing and blur
# ROI = image[y:y+h, x:x+w]
# blur = cv2.GaussianBlur(ROI, (51,51), 0)
#
# # Insert ROI back into image
# image[y:y+h, x:x+w] = blur
#
# cv2.imshow('blur', blur)
# cv2.imshow('image', image)
# cv2.waitKey()


import cv2
import numpy as np
import Hand_Tracking_Module as htm

cap = cv2.VideoCapture(0)
detector = htm.HandDetector()


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    lmlist = detector.findPosition( img , draw = True)


    # for i in range(0,len(img)):
    #     img = img[i]
        # img = cv2.imread("lena.jpg")
    blurred_img = cv2.GaussianBlur(img, (91,91), 0)

    mask = np.zeros((480,640, 3), dtype=np.uint8)
    color = np.random.randint(low=255, high=256, size=3).tolist()
    print(color)
    # mask = cv2.circle(mask, (258, 258), 10, color, -1)
    mask = cv2.rectangle(mask ,(250,250) , (500,500) , color , -1 )

    img = np.where(mask==color, img, blurred_img)

    # cv2.imwrite("./out.png", out)
    cv2.imshow("Image", img)
    cv2.waitKey(1)