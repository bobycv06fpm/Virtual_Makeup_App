import cv2 as cv
from imutils.video import WebcamVideoStream
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

class VideoCamera(object):
    def __init__(self):
        self.stream = WebcamVideoStream(src=0).start()

    def get_frame(self):
        image = self.stream.read()

        imgOriginal = image.copy()
        imgGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        mask = np.zeros_like(image)
        faces = detector(imgGray)

        for face in faces:  # 얼굴 인식
            # x1, y1 = face.left(), face.top()
            # x2, y2 = face.right(), face.bottom()
            # imgOriginal = cv.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
            landmarks = predictor(imgGray, face)

            myPoints = []
            for n in range(68):  # 이목구비 인식
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                myPoints.append([x, y])
                cv.circle(image, (x, y), 2, (50, 50, 255), cv.FILLED)
                # cv.putText(imgOriginal, str(n), (x, y - 10), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1)

        ret, jpeg = cv.imencode('.jpg', image)
        data = []
        data.append(jpeg.tobytes())
        return data