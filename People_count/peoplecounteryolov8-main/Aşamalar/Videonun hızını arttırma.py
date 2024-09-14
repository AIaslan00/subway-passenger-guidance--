import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8s.pt')

area1 = [(312, 388), (289, 390), (474, 469), (497, 462)]
area2 = [(279, 392), (250, 397), (423, 477), (454, 469)]

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('C:/Users/Aslan/OneDrive/Masaüstü/Bitirme Projesii/peoplecount1/peoplecount1.mp4')

my_file = open("C:/Users/Aslan/OneDrive/Masaüstü/Bitirme Projesii/peoplecounteryolov8-main/coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

skip_frames = 3  # Her bir çerçeveden kaç tanesini atlayacağınızı belirleyin

while True:
    ret, frame = cap.read()
    if not ret:
        break
    for _ in range(skip_frames - 1):
        cap.read()  # Belirli sayıda çerçeveyi atla
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'person' in c:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255, 255, 255), 1)

    cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, str('1'), (504, 471), cv2.FONT_HERSHEY_COMPLEX, (0.5), (0, 0, 0), 1)

    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, str('2'), (466, 485), cv2.FONT_HERSHEY_COMPLEX, (0.5), (0, 0, 0), 1)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
