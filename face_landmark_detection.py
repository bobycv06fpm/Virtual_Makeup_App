import cv2 as cv
import dlib
from threading import Thread

#   이목구비 지정
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INNER = list(range(61, 68))
JAWLINE = list(range(0, 17))
LEFT_CHEEK = [4, 3, 2, 1, 31, 50, 49, 48]
RIGHT_CHEEK = [12, 13, 14, 15, 35, 52, 53, 54]

# RIGHT_CHEEK = list(range(29, 33)) + list(range(12, 54))
# LEFT_CHEEK = list(range(29, 33)) + list(range(4, 48))

class Webcam:
    def __init__(self):
        self.data = None
        self.cam = cv.VideoCapture(0)

        self.WIDTH = 640
        self.HEIGHT = 480

        self.center_x = self.WIDTH / 2
        self.center_y = self.HEIGHT / 2

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")

        self.myPoints = []

    def stream(self):
        #   스트리밍하며 이목구비 검출하는 함수
        def streaming():
            while True:
                self.ret, img = self.cam.read()
                if img is None:
                    continue

                #   거울 모드 좌우 반전
                img = cv.flip(img, 1)

                img = cv.resize(img, (0, 0), None, 0.9, 0.9)
                imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                faces = self.detector(imgGray)

                for face in faces:
                    #   얼굴 및 이목구비 인식
                    x1, y1 = face.left(), face.top()
                    x2, y2 = face.right(), face.bottom()
                    imgOriginal = cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    landmarks = self.predictor(imgGray, face)

                    self.myPoints = []
                    for n in range(68):
                        x = landmarks.part(n).x
                        y = landmarks.part(n).y
                        self.myPoints.append([x, y])
                        cv.circle(imgOriginal, (x, y), 2, (50, 50, 255), cv.FILLED)
                        # cv.putText(imgOriginal, str(n), (x, y - 10), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1)

                self.data = img

                k = cv.waitKey(1)
                if k == ord('q'):
                    self.release()
                    break

        Thread(target=streaming).start()

    def show(self):
        # 보여주는 함수
        while True:
            frame = self.data
            if frame is not None:
                cv.imshow('frame', frame)

            key = cv.waitKey(1)
            if key == 27:
                self.release()
                cv.destroyAllWindows()
                break

    def release(self):
        self.cam.release()
        cv.destroyAllWindows()

if __name__ == '__main__':
    cam = Webcam()
    cam.stream()
    cam.show()