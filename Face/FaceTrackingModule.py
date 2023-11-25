import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, min_detection_confidence=0.85, model_selection=0):
        self.detectionConf = min_detection_confidence
        self.model_selection = model_selection
        
        self.mpFace = mp.solutions.face_detection
        self.face = self.mpFace.FaceDetection(min_detection_confidence=self.detectionConf, model_selection=self.model_selection)
        self.mpDraw = mp.solutions.drawing_utils


    def findFace(self, img, draw=True):
        imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face.process(imageRGB)
        
        if self.results.detections:
            for i, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                        int(bboxC.width * iw), int(bboxC.height * ih)
                if draw:
                    cv2.rectangle(img, bbox, (255, 0, 255), 2)
                    cv2.putText(img, str(f'{int(detection.score[0] * 100)}%'), (bbox[0],bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
                    
        return img
    
    # def findPosition(self, img, handNo=0, draw=True):
    #     lmList = []
    #     if self.results.face_landmarks:
    #         myFace = self.results.face_landmarks
            
    #         for id, lm in enumerate(myFace.landmark):
    #             h,w,c = img.shape
    #             cx, cy = int(lm.x*w), int(lm.y*h)
    #             print(id, cx, cy)
    #             lmList.append([id,cx,cy])
    #             if draw:
    #                 cv2.circle(img, (cx,cy), 15, (255,0,0), cv2.FILLED)

    #     return lmList

def main():
    pTime = 0
    cTime = 0
    
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img = detector.findFace(img)
        # lmList = detector.findPosition(img)

        # if len(lmList) != 0:
        #     print(lmList[4])

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
        
if __name__ == "__main__":
    main()