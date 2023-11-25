import cv2
import time
import numpy as np
import os
import sys
import math

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

import Hands.HandTrackingModule as htm

def initialize_camera(width, height):
    """ Inicializa la cámara con el ancho y alto especificados. """
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)
    return cap

def main():
    """ Función principal para controlar el volumen mediante la detección de manos. """
    camera_width, camera_height = 640, 480
    pTime = 0
    vol = 0
    volBar = 0

    cap = initialize_camera(camera_width, camera_height)
    detector = htm.HandDetector(detectionCon=0.7)

    while True:
        success, img = cap.read()
        if not success:
            break

        detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            x1, y1 = lmList[4][1], lmList[4][2] # Thumb tip
            x2, y2 = lmList[8][1], lmList[8][2] # Index finger tip

            cx, cy = (x1+x2)//2, (y1+y2)//2


            cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)

            length = math.hypot(x2-x1, y2-y1)
            vol = np.interp(length, [25, 200], [0, 100])
            volBar = np.interp(length, [25, 200], [400, 150])
            applescript_command = "osascript -e 'set volume output volume {}'".format(vol)
            os.system(applescript_command)

            if length < 25:
                cv2.circle(img, (cx, cy), 7, (0, 255, 0), cv2.FILLED)

        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, f"{int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
        
if __name__ == "__main__":
    main()