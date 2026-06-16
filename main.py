from util import *
from sort.sort import *
import time

mot_tracker = Sort()
car_readings = {}
car_best = {}
car_locked = {}
last_plate_crop = None
frame_nmr = -1
last_plate_time = time.time()

cap = cv2.VideoCapture('./video_auta3.mp4')
ret = True

while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if not ret:
        break

    frame, car_readings, car_best, car_locked, last_plate_crop, last_plate_time, sorted_spz = \
        process_frame(frame, mot_tracker, car_readings, car_best, car_locked, last_plate_crop, last_plate_time)
    
    cv2.imshow('YOLO + SPZ (5 enginu)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print('\n[INFO] Hotovo.')
for txt, count in sorted_spz:
    print(f'  {txt}  ({count}x)')