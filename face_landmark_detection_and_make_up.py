"""
고칠 것
눈썹 이어지는 거 수정

추가할 것
볼터치 같은 영역 구할 수 있을 거 같은데?
https://stackoverflow.com/questions/54300968/how-to-detect-cheeks-using-opencv

되면 할 것
좀 더 화장처럼 어떻게 하면 좋을까? 진하기 이런 거... 이거는 인공지능 영역인가?
"""


import cv2 as cv
import numpy as np
import dlib
from imutils import face_utils

webcam = True

cap = cv.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#이목구비 지정
ALL = list(range(0, 68))
RIGHT_EYEBROW = list(range(17,22))
LEFT_EYEBROW = list(range(22,27))
RIGHT_EYE = list(range(36,42))
LEFT_EYE = list(range(42,48))
NOSE = list(range(27,36))
MOUTH_OUTLINE = list(range(48,61))
MOUTH_INNER = list(range(61,68))
JAWLINE = list(range(0,17))
index = ALL
index2 = RIGHT_EYEBROW

# 트랙바 설정
def empty(a):
    pass

cv.namedWindow("BGR")
cv.resizeWindow("BGR", 640, 240)
cv.createTrackbar("Blue", 'BGR', 0, 255, empty)
cv.createTrackbar("Green", 'BGR', 0, 255, empty)
cv.createTrackbar("Red", 'BGR', 0, 255, empty)

def createBox(img, points, scale=5, masked=False, cropped=True): # 얼굴에서 입력된 부분만 잘라내는 함수
    if masked:
        mask = np.zeros_like(img)
        mask = cv.fillPoly(mask, [points], (255,255,255)) # 정확한 영역 추출을 위해 rectangle 말고 fillpoly 사용
        img = cv.bitwise_and(img, mask)
        #cv.imshow('Mask', img)
    if cropped:
        bbox = cv.boundingRect(points) # x, y, 넓이, 높이 반환
        x,y,w,h = bbox
        imgCrop = img[y:y+h, x:x+w]
        imgCrop =cv.resize(imgCrop, (0,0), None, scale, scale)
        return imgCrop
    else:
        return mask

while True:
    if webcam: success, img = cap.read()
    else: img = cv.imread('face.jpg')

    #img = cv.resize(img, (0,0), None, 0.9, 0.9)
    imgOriginal = img.copy()
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = detector(imgGray)

    for face in faces: # 얼굴 인식
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        imgOriginal = cv.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        landmarks = predictor(imgGray, face)

        myPoints = []
        for n in range(68): # 얼굴에서 이목구비 인식
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            myPoints.append([x,y]) # 68개 포인트 좌표값 저장
            #cv.circle(imgOriginal, (x,y), 3, (50,50,255), cv.FILLED)
            #cv.putText(imgOriginal, str(n), (x,y-10), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,255), 1)

        # 이목구비 영역 지정
        myPoints = np.array(myPoints) # 넘파이 배열로 변경
        imgFeatures = createBox(img, myPoints[index], 3, masked=True, cropped=False)

        # 이목구비 초기화
        imgColorFeatures = np.zeros_like(imgFeatures)

        # 트랙바
        b = cv.getTrackbarPos('Blue', 'BGR')
        g = cv.getTrackbarPos('Green', 'BGR')
        r = cv.getTrackbarPos('Red', 'BGR')

        # 선택된 영역 색칠
        imgColorFeatures[:] = b, g, r
        imgColorFeatures = cv.bitwise_and(imgFeatures, imgColorFeatures)
        imgColorFeatures = cv.GaussianBlur(imgColorFeatures, (7,7), 10)
        #imgOriginalGray = cv.cvtColor(imgOriginal, cv.COLOR_BGR2GRAY) # 채널 똑같이 하기 위해서 조정
        #imgOriginalGray = cv.cvtColor(imgOriginalGray, cv.COLOR_GRAY2BGR) # 채널 똑같이 하기 위해서 조정
        imgColorFeatures = cv.addWeighted(imgOriginal, 1, imgColorFeatures, 0.4, 0)
        cv.imshow('BGR', imgColorFeatures)
        cv.imshow('Lips', imgFeatures)

        #print(myPoints)


    cv.imshow("Original", imgOriginal)
    key = cv.waitKey(1)
    if key == 27:
        break
    elif key == ord('1'):
        index = ALL
    elif key == ord('2'):
        index = LEFT_EYEBROW and RIGHT_EYEBROW
    elif key == ord('3'):
        index = LEFT_EYE + RIGHT_EYE
    elif key == ord('4'):
        index = NOSE
    elif key == ord('5'):
        index = MOUTH_OUTLINE + MOUTH_INNER
    elif key == ord('6'):
        index = JAWLINE