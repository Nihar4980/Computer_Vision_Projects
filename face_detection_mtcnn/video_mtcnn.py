from mtcnn import MTCNN
import cv2

detector = MTCNN()

cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    output = detector.detect_faces(frame)
    print(output)
    for i in output:
        x,y,width,height = i['box']

        left_eyeX, left_eyeY = i['keypoints']['left_eye']
        right_eyeX, right_eyeY = i['keypoints']['right_eye']
        noseX, noseY = i['keypoints']['nose']
        mouth_leftX, mouth_leftY = i['keypoints']['mouth_left']
        mouth_rightX, mouth_rightY = i['keypoints']['mouth_right']


        cv2.circle(frame,center = (left_eyeX, left_eyeY),color = (0,255,0),thickness=3,radius=2)
        cv2.circle(frame,center = (right_eyeX, right_eyeY),color = (0,255,0),thickness=3,radius=2)
        cv2.circle(frame,center = (noseX, noseY),color = (0,255,0),thickness=3,radius=2)
        cv2.circle(frame,center = (mouth_leftX, mouth_leftY),color = (0,255,0),thickness=3,radius=2)
        cv2.circle(frame,center = (mouth_rightX, mouth_rightY),color = (0,255,0),thickness=3,radius=2)

        cv2.rectangle(frame,pt1 = (x,y), pt2 = (x+width,y+height),color=(255,0,0),thickness=2)
    cv2.imshow('window',frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cv2.destroyAllWindows()
