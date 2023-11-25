import cv2
import mediapipe as mp
import time

class PoseDetectior():
    def __init__(self,
               static_image_mode=False,
               model_complexity=1,
               smooth_landmarks=True,
               enable_segmentation=False,
               smooth_segmentation=True,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
        self.mode = static_image_mode
        self.complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detectionConf = min_detection_confidence
        self.trackConf = min_tracking_confidence
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
               model_complexity=self.complexity,
               smooth_landmarks=self.smooth_landmarks,
               enable_segmentation=self.segmentation,
               smooth_segmentation=self.smooth_segmentation,
               min_detection_confidence=self.detectionConf,
               min_tracking_confidence=self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils
        

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
            
        return img
    
    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            myPose = self.results.pose_landmarks
            
            for id, lm in enumerate(myPose.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)

        return lmList
        

def main():
    pTime = 0
    cTime = 0
    
    cap = cv2.VideoCapture(0)
    detector = PoseDetectior()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img)

        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
    
    
if __name__ == '__main__':
    main()