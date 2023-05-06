from ultralytics import YOLO
import cv2

model = YOLO('../Yolo-Weights/yolov8n-seg.pt')
results = model("Images/bus.jpg", show=True)
cv2.waitKey(0)