import cv2
import dlib
from math import hypot
import time
import threading
from playsound import playsound

cap = cv2.VideoCapture(0)
#startTime is for keeping the starting time of the program.
startTime= int(round(time.time() * 1000))
soundTime=0
once=0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def middle(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_SIMPLEX

#function for calculating one eye's horizontal length divided by vertical length.
def getRatio(eye_points, facial_landmarks):
    #fing left,right,top,bottom points of the eye to measure the distance
    left = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    topMid = middle(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    bottomMid = middle(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    #show those vertical and horizontal lines in the video.
    horizontalLine = cv2.line(frame, left, right, (0, 128, 255), 1)
    verticalLine = cv2.line(frame, topMid, bottomMid, (0, 128, 255), 1)

    #calculate the lengths and find ratio
    horizontalLineLength = hypot((left[0] - right[0]), (left[1] - right[1]))
    verticalLineLength = hypot((topMid[0] - bottomMid[0]), (topMid[1] - bottomMid[1]))

    ratio = horizontalLineLength / verticalLineLength
    return ratio

#function for giving the sound with threading
def voice():
    playsound('smashingSound.wav')

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmark = predictor(gray, face)
        #keep track of the time to determine when sound is going to be provided
        now = int(round(time.time() * 1000))
        leftEye = getRatio([36, 37, 38, 39, 40, 41], landmark)
        rightEye = getRatio([42, 43, 44, 45, 46, 47], landmark)
        blinkRatio = (leftEye+ rightEye) / 2

        #creating the thread for sound giving
        t1 = threading.Thread(target=voice, args=())
        #if ((now > counter + 4) and (now < counter + 1000 * 4, 5)):

        #determine when the sound is going to be given . In this case it is 6 seconds after start time.
        if ((now > startTime + 6) ):
            if (once == 0): #for making sure give the sound just one time.
                soundTime = now
                once =once + 1
                t1.start()

        if blinkRatio > 5.7:
            blinkTime=int(round(time.time() * 1000))
            delayTime= blinkTime - soundTime #calculate eye response delay
            cv2.imwrite("frame %d.png"  % delayTime, frame) #save the frames including blinks to the folder.
            print("You blinked at: ",blinkTime)
            print("Delay is: ",delayTime ," milliseconds")

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
