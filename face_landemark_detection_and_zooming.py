import cv2 as cv
import numpy as np
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
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        self.scale = 1
        self.area = 0

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

                    myPoints = []
                    for n in range(68):
                        x = landmarks.part(n).x
                        y = landmarks.part(n).y
                        myPoints.append([x, y])
                        cv.circle(imgOriginal, (x, y), 2, (50, 50, 255), cv.FILLED)
                        # cv.putText(imgOriginal, str(n), (x, y - 10), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1)

                #   줌 영역 분기
                if self.area == 1:
                    # 왼쪽 눈&눈썹
                    point = (np.array(myPoints[39]) + np.array(myPoints[36])) / 2
                    img = self.__zoom(img, (int(point[0]), int(point[1])))
                elif self.area == 2:
                    # 오른쪽 눈&눈썹
                    point = (np.array(myPoints[45]) + np.array(myPoints[42])) / 2
                    img = self.__zoom(img, (int(point[0]), int(point[1])))
                elif self.area == 3:
                    # 코
                    point = (np.array(myPoints[30]) + np.array(myPoints[27])) / 2
                    img = self.__zoom(img, (int(point[0]), int(point[1])))
                elif self.area == 4:
                    # 왼쪽 볼
                    point = (np.array(myPoints[5]) + np.array(myPoints[36])) / 2
                    img = self.__zoom(img, (int(point[0]), int(point[1])))
                elif self.area == 5:
                    # 오른쪽 볼
                    point = (np.array(myPoints[11]) + np.array(myPoints[45])) / 2
                    img = self.__zoom(img, (int(point[0]), int(point[1])))
                elif self.area == 6:
                    # 입술
                    point = (np.array(myPoints[54]) + np.array(myPoints[48])) / 2
                    img = self.__zoom(img, (int(point[0]), int(point[1])))

                if self.area == 0:
                    img = self.__zoom(img, (self.center_x, self.center_y))

                self.data = img

                k = cv.waitKey(1)
                if k == ord('q'):
                    self.release()
                    break

        Thread(target=streaming).start()

    def __zoom(self, img, center=None):
        #   zoom하는 실제 함수
        height, width = img.shape[:2]
        if center is None:
            #   중심값이 초기값일 때의 계산
            center_x = int(width / 2)
            center_y = int(height / 2)
            radius_x, radius_y = int(width / 2), int(height / 2)
        else:
            #   특정 위치 지정시 계산
            rate = height / width
            center_x, center_y = center

            #   비율 범위에 맞게 중심값 계산
            if center_x < width * (1 - rate):
                center_x = width * (1 - rate)
            elif center_x > width * rate:
                center_x = width * rate
            if center_y < height * (1 - rate):
                center_y = height * (1 - rate)
            elif center_y > height * rate:
                center_y = height * rate

            center_x, center_y = int(center_x), int(center_y)
            left_x, right_x = center_x, int(width - center_x)
            up_y, down_y = int(height - center_y), center_y
            radius_x = min(left_x, right_x)
            radius_y = min(up_y, down_y)

            #   실제 zoom 코드
        radius_x, radius_y = int(self.scale * radius_x), int(self.scale * radius_y)

        #   size 계산
        min_x, max_x = center_x - radius_x, center_x + radius_x
        min_y, max_y = center_y - radius_y, center_y + radius_y

        #   size에 맞춰 이미지를 자른다
        cropped = img[min_y:max_y, min_x:max_x]
        #   원래 사이즈로 늘려서 리턴
        new_cropped = cv.resize(cropped, (width, height))

        return new_cropped

    def zoom_init(self):
        self.center_x = self.WIDTH / 2
        self.center_y = self.HEIGHT / 2
        self.scale = 1
        self.area = 0

    def zoom_in(self):
        if self.scale > 0.2:
            self.scale -= 0.1

    def show(self):
        # 보여주는 함수
        while True:
            frame = self.data
            if frame is not None:
                cv.imshow('frame', frame)

            key = cv.waitKey(1)
            if key == ord('q'):
                self.release()
                cv.destroyAllWindows()
                break
            elif key == ord('z'):
                # 중앙 줌인
                self.zoom_in()
            elif key == ord('1'):
                # 왼쪽 눈&눈썹 줌인
                self.area = 1
                self.zoom_in()
            elif key == ord('2'):
                # 오른쪽 눈&눈썹 줌인
                self.area = 2
                self.zoom_in()
            elif key == ord('3'):
                # 코 줌인
                self.area = 3
                self.zoom_in()
            elif key == ord('4'):
                # 왼쪽 볼 줌인
                self.area = 4
                self.zoom_in()
            elif key == ord('5'):
                # 오른쪽 볼 줌인
                self.area = 5
                self.zoom_in()
            elif key == ord('6'):
                # 입술 줌인
                self.area = 6
                self.zoom_in()
            elif key == ord('v'):
                # 원상태로 복구
                self.zoom_init()

    def release(self):
        self.cam.release()
        cv.destroyAllWindows()

if __name__ == '__main__':
    cam = Webcam()
    cam.stream()
    cam.show()