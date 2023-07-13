from ultralytics import YOLO
import cv2
import cvzone
import math

# cap = cv2.VideoCapture(1)  # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)
#cap = cv2.VideoCapture("../Videos/ppe-2.mp4")  # For Video
cap = cv2.imread("drone1.jpg")

model = YOLO("drone.pt")

classNames = ['Autel_EVO_2', 'DJI_FPV', 'DJI_MINI_2', 'DJI_Phantom_4', 'P_P_X_Wizard', 'Ryze_Tello']

bBoxColor = (0, 0, 255)
while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            # cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            print(currentClass)
            if conf>0.5:

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1,colorB=bBoxColor,
                                   colorT=(255,255,255),colorR=bBoxColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), bBoxColor, 3)


    cv2.imshow("Image", img)
    cv2.waitKey(1)
