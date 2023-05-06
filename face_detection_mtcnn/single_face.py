from mtcnn import MTCNN
import cv2

detector = MTCNN()

img = cv2.imread('images/vk.jpg')

output = detector.detect_faces(img)
print(output)

x,y,width,height = output[0]['box']

left_eyeX, left_eyeY = output[0]['keypoints']['left_eye']
right_eyeX, right_eyeY = output[0]['keypoints']['right_eye']
noseX, noseY = output[0]['keypoints']['nose']
mouth_leftX, mouth_leftY = output[0]['keypoints']['mouth_left']
mouth_rightX, mouth_rightY = output[0]['keypoints']['mouth_right']


cv2.circle(img,center = (left_eyeX, left_eyeY),color = (0,255,0),thickness=3,radius=2)
cv2.circle(img,center = (right_eyeX, right_eyeY),color = (0,255,0),thickness=3,radius=2)
cv2.circle(img,center = (noseX, noseY),color = (0,255,0),thickness=3,radius=2)
cv2.circle(img,center = (mouth_leftX, mouth_leftY),color = (0,255,0),thickness=3,radius=2)
cv2.circle(img,center = (mouth_rightX, mouth_rightY),color = (0,255,0),thickness=3,radius=2)

cv2.rectangle(img,pt1 = (x,y), pt2 = (x+width,y+height),color=(255,0,0),thickness=2)
cv2.imshow('window',img)

cv2.waitKey(0)

