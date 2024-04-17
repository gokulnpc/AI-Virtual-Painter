import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

cTime = 0
pTime = 0

detector = htm.handDetector(detectionCon=0.75, maxHands=1)

drawPoints = []
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # index finger tip
        x1, y1 = lmList[8][1:]
        # apply to list
        drawPoints.append((x1, y1))
    
    # draw points
    for point in drawPoints:
        cv2.circle(img, point, 5, (255, 0, 255), cv2.FILLED)
    
    # Erase
    # two fingers up
    if len(lmList) != 0:
        fingers = detector.fingersUp()
        if fingers[1] and fingers[2]:
            drawPoints = []
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
    