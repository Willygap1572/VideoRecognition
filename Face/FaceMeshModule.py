import cv2
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self,
               static_image_mode=False,
               max_num_faces=1,
               refine_landmarks=False,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
        
        self.mode = static_image_mode
        self.maxFaces = max_num_faces
        self.refine_landmarks= refine_landmarks
        self.detectionConf = min_detection_confidence
        self.trackConf = min_tracking_confidence
        
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
               static_image_mode=self.mode,
               max_num_faces=self.maxFaces,
               refine_landmarks=self.refine_landmarks,
               min_detection_confidence=self.detectionConf,
               min_tracking_confidence=self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)


    def findFaceMesh(self, img, draw=True):
        self.imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imageRGB)
        
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION,
                                               self.drawSpec, self.drawSpec)
                face = []
                for id,lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)                    
                    face.append([x, y])
                faces.append(face)
        return img, faces

def main():
    pTime = 0
    cTime = 0
    
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        
        if len(faces) != 0:
            print(len(faces))
        
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
        
if __name__ == "__main__":
    main()