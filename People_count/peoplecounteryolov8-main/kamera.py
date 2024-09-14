import cv2
import pandas as pd
import numpy as np
import mysql.connector
from ultralytics import YOLO
from tracker import *

# Kameradan görüntü almak için 0 indeksini kullanıyoruz
cap = cv2.VideoCapture(0)

# Class list tanımlamasını buraya ekleyin
my_file = open("C:/Users/Aslan/OneDrive/Masaüstü/Bitirme Projesii/peoplecounteryolov8-main/coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# MySQL bağlantısı
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="AI023",
    database="metro"
)

# Veritabanı bağlantısı kontrolü
if mydb.is_connected():
    print("MySQL veritabanına başarıyla bağlanıldı.")
else:
    print("Veritabanına bağlantı başarısız.")

# Cursor oluşturma
mycursor = mydb.cursor()

# Tabloya veri ekleme fonksiyonu
def insert_data(entering, exiting, vagon):
    sql = "INSERT INTO metro_data (entering_count, exiting_count, vagon_count) VALUES (%s, %s, %s)"
    val = (entering, exiting, vagon)
    mycursor.execute(sql, val)
    mydb.commit()
    print(mycursor.rowcount, "kayıt eklendi.")

# Veritabanından veri çekme fonksiyonu
def fetch_data():
    mycursor.execute("SELECT entering_count, exiting_count, vagon_count FROM metro_data")
    result = mycursor.fetchall()
    return result

model = YOLO('yolov8s.pt')

# Giriş/çıkış alanları ve diğer tanımlamaları buraya ekleyin
area1 = [(90, 460), (90, 420), (640, 420), (640, 460)]
area2 = [(90, 400), (90, 360), (640, 360), (640, 400)]

count = 0
tracker = Tracker()
entering = set()
people_entering = {}
people_exiting = {}
exiting = set()

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (255, 255, 255)
thickness = 2

vagon = 1  # Başlangıçta vagon 1'de başlayalım

# MySQL veritabanına ekleme işlemini güncelleme fonksiyonu
def update_database(entering_count, exiting_count, vagon_count):
    sql = "INSERT INTO metro_data (entering_count, exiting_count, vagon_count) VALUES (%s, %s, %s)"
    val = (entering_count, exiting_count, vagon_count)
    mycursor.execute(sql, val)
    mydb.commit()
    print(mycursor.rowcount, "kayıt güncellendi.")

# Ana döngü
while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 2 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    bbox_list = []
    
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'person' in c:
            bbox_list.append([x1, y1, x2, y2])
    
    bbox_id = tracker.update(bbox_list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        results = cv2.pointPolygonTest(np.array(area2, np.int32), ((x4, y4)), False)
        if results >= 0:
            people_entering[id] = (x4, y4)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 0), 2)
        
        if id in people_entering:
            results1 = cv2.pointPolygonTest(np.array(area1, np.int32), ((x4, y4)), False)
            if results1 >= 0:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                cv2.circle(frame, (x4, y4), 5, (255, 0, 255), 1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255, 255, 255), 1)
                entering.add(id)

        results2 = cv2.pointPolygonTest(np.array(area1, np.int32), ((x4, y4)), False)
        if results2 >= 0:
            people_exiting[id] = (x4, y4)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
        
        if id in people_exiting:
            results3 = cv2.pointPolygonTest(np.array(area2, np.int32), ((x4, y4)), False)
            if results3 >= 0:
                cv2.rectangle(frame, (x3,y3), (x4, y4), (0, 0, 255), 2)
                cv2.circle(frame, (x4, y4), 5, (0, 0, 255), 1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255, 255, 255), 1)
                exiting.add(id)

    cv2.polylines(frame, [np.array(area1, np.int32)], True, (150, 150, 150), 2)
    cv2.putText(frame, str('1'), (120, 440), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 1)

    cv2.polylines(frame, [np.array(area2, np.int32)], True, (150, 150, 150), 2)
    cv2.putText(frame, str('2'), (120, 380), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 1)

    # Yoğunluk hesaplama
    total_capacity = 20  # Bir vagonun toplam kapasitesi
    current_capacity = len(entering) - len(exiting)

    # Çıkan sayısı giren sayısından büyükse yoğunluk yüzdesini 0 yap
    if current_capacity < 0:
        density_percentage = 0
    else:
        density_percentage = int((current_capacity / total_capacity) * 100)

    capacity_difference = len(entering) - len(exiting)
    # Vagon sayısını güncelle
    if capacity_difference > 3:
        vagon = min(2 + ((capacity_difference - 3) // 3), 6)  # Vagon sayısını 6'dan fazla yapma
        if vagon == 1:
            cv2.putText(frame, "Vagon 2'ye gidebilirsiniz.", (640, 150), font, fontScale, color, thickness)
        elif vagon == 2:
            cv2.putText(frame, "Vagon 2'e gidebilirsiniz.", (640, 150), font, fontScale, color, thickness)
        elif vagon == 3:
            cv2.putText(frame, "Vagon 3'e gidebilirsiniz.", (640, 150), font, fontScale, color, thickness)
        elif vagon == 4:
            cv2.putText(frame, "Vagon 4'e gidebilirsiniz.", (640, 150), font, fontScale, color, thickness)
        elif vagon == 5:
            cv2.putText(frame, "Vagon 5'ya gidebilirsiniz.", (640, 150), font, fontScale, color, thickness)
        elif vagon == 6:
            cv2.putText(frame, "Vagon 6'ya gidebilirsiniz", (640, 150), font, fontScale, color, thickness)

    elif capacity_difference <= 0:
        density_percentage = 0  # Yoğunluk yüzdesi 0
        vagon = 1  # Çıkan sayısı giren sayısından büyükse vagon 1'e geç

    # Yoğunluk yüzdesini ekrana yazdır
    cv2.putText(frame, f"Yogunluk: %{density_percentage}", (700, 50), font, fontScale, (200,200,0), thickness)
    
    # Vagon sayısını ekrana yazdır
    cv2.putText(frame, f"Vagon: {vagon}", (700, 100), font, fontScale, (200,0,200), thickness)

    cv2.putText(frame, f"Giren: {len(entering)}", (50, 50), font, fontScale, (0, 255, 0), thickness)
    cv2.putText(frame, f"Cikan: {len(exiting)}", (50, 100), font, fontScale, (0, 0, 255), thickness)

    cv2.imshow("Metro Giris", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # Her karede veritabanını güncelle
    update_database(len(entering), len(exiting), vagon)

cap.release()
cv2.destroyAllWindows()
# Veritabanından verileri çekme ve yazdırma
data = fetch_data()
for row in data:
    print(f"Giren: {row[0]}, Çıkan: {row[1]}, Vagon: {row[2]}")

