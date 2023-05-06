from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture(0)

cap.set(3,1440)
cap.set(4,900)


model = YOLO('../Yolo-Weights/yolov8x-seg.pt')
model.predict(source='0',show=True)