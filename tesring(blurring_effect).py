import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    # h, w, c = img.shape
    # self.bbox = [x_max, y_max, x_min, y_min]
    # print(self.bbox)
    # Create ROI coordinates
    # topLeft = (x_min, y_min)
    # bottomRight = (x_max, y_max)
    # x, y = self.bbox[0][2], self.bbox[0][3]
    # lw, lh = self.bbox[0][0] - self.bbox[0][2], self.bbox[0][1] - self.bbox[0][3]

    # if len(self.bbox) != 0:
    # x, y = self.bbox[0][2], self.bbox[0][3]
    # lw, lh = self.bbox[0][0], self.bbox[0][1]

    # Grab ROI with Numpy slicing and blur
    # ROI = img[0:h ,0:w ] - img[y:y + lh, x:x + lw]
    # ROI = img[y:y + lh, x:x + lw]
    blur = cv2.GaussianBlur(img, (151, 151), 0)

    # Insert ROI back into image
    # img[y:y + lh, x:x + lw] = blur

    mask = np.zeros((480, 640, 3), dtype=np.uint8)
    color = np.random.randint(low=255, high=256, size=3).tolist()
    print(color)
    mask = cv2.rectangle(mask,(100,100) , (200,200), color, -1)
    img = np.where(mask == color, img, blur)
    cv2.imshow("Blurred" , img)
    cv2.waitKey(1)