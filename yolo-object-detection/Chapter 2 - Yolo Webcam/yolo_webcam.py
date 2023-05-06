from ultralytics import YOLO
import cv2
import cvzone

cap = cv2.VideoCapture(0)

cap.set(3,1440)
cap.set(4,900)


model = YOLO('../Yolo-Weights/yolov8n-seg.pt')

while True:
    sucess, img = cap.read()
    results = model(img,stream=True)
    #print(results)
    for r in results:
        print(r)
        boxes = r.boxes
        masks = r.masks

        for box,mask in zip(boxes,masks):
            x1,y1,x2,y2 = box.xyxy[0]

            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            print(x1,y1,x2,y2)

            print(mask)

            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)

    cv2.imshow('Image',img)
    cv2.waitKey(1)
