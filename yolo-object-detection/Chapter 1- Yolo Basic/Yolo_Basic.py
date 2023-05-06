from ultralytics import YOLO
import cv2

model = YOLO('../Yolo-Weights/yolov8n.pt')
results = model("../Chapter 1- Yolo Basic/video_1.mp4", show=True)
cv2.waitKey(1)