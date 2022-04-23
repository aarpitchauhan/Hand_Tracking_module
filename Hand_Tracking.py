import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

mpHands = mp.solutions.hands    #detect the hands, in default ( look inside the class “Hands()“)
hands = mpHands.Hands()   #uses only RGB images
mpDraw = mp.solutions.drawing_utils #to draw the key points


pTime = 0   #past time
cTime = 0   #current time


while True:
    success , img = cap.read()
    imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)  #convert the image to RGB
    results = hands.process(imgRGB)  # detect hands in the frame
    # print(results.multi_hand_landmarks)  #this is to verify whether the hand is detected or not

    '''Once the hands get detected we will locate the key points 
    and then we  highlight the dots in the keypoints using cv2.circle,
    and connect the key points using mpDraw.draw_landmarks.'''

    if results.multi_hand_landmarks:  # contains the hand landmarks on each detected hand (refer 'hands' in mediapipe line documentation)
        for handLms in results.multi_hand_landmarks:
            for id , lm in enumerate(handLms.landmark):
                # print(id , lm) #here id refers to a point out of 21 points and lm refers to landmark(x,y,z) coordinate of that point
                #above calculated landmarks are in decimals
                #so we will convert in form of pixels such that we can locate it in picture using coordinate of pixels

                #code for converting decimal landmarks into pixel type coordinate system
                h , w , c = img.shape  #checking hight , width and channels of the image
                cx , cy = int(lm.x*w) , int(lm.y*h)  #finding positons of center
                # print(id , cx , cy)
                # if id == 0:  #every point on hand have some ID so by specifying point ID we can locate a definite point
                    # cv2.circle(img , (cx,cy) , 15 , (255,0,255) , -1 )  #creating circle to highlight keypoints



            mpDraw.draw_landmarks(img , handLms , mpHands.HAND_CONNECTIONS) #here mpHands.HAND_CONNECTIONS connecting landmark points



    # To find FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img , str(int(fps)) , (18,70), cv2.FONT_HERSHEY_SIMPLEX ,
                3 , (255 , 0 ,255) , 3)

    cv2.imshow("Image" , img )
    # --------------------------------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



