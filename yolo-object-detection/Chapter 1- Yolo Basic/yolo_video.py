from ultralytics import YOLO
import cv2

model = YOLO('../Yolo-Weights/yolov8x-seg.pt')
model.predict("C:/Users/Nihar/Downloads/pexels-mike-bird-2053100-3840x2160-60fps.mp4", show=True)
cv2.waitKey(1)