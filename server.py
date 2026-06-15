from util import *
from sort.sort import Sort
from flask import Flask, Response, jsonify, render_template, request
import time
import threading

suppress_warnings()

app = Flask(__name__)

# sdileny stav mezi vlakny
lock = threading.Lock()
current_frame_jpg = None
current_spz_list = []  # [{"text": "1AB 1234", "count": 5}, ...]
is_running = False

# --- DETEKCE VLAKNO ---

def detection_loop():
    global current_frame_jpg, current_spz_list, is_running
    try:
        _run_detection()
    except Exception as e:
        print(f'[ERROR] Detection crashed: {e}')
        import traceback
        traceback.print_exc()

def _run_detection():
    global current_frame_jpg, current_spz_list, is_running

    # --- HLAVNI SMYCKA ---
    mot_tracker = Sort()

    cap = cv2.VideoCapture(source)
    car_readings = {}
    car_best = {}
    car_locked = {}  # car_id -> True kdyz uz ma finalní SPZ
    last_plate_crop = None  # posledni vyrez SPZ pro zobrazeni
    frame_nmr = -1
    last_plate_time = time.time()
    is_running = True

    while True:
        frame_nmr += 1
        ret, frame = cap.read()
        if not ret:
            # restartuj video od zacatku
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_nmr = -1
            car_readings.clear()
            car_best.clear()
            mot_tracker = Sort()
            continue

        # --- DETEKCE AUT ---
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        if len(detections_) > 0:
            track_ids = mot_tracker.update(np.asarray(detections_))
        else:
            track_ids = mot_tracker.update(np.empty((0, 5)))

        # --- DETEKCE SPZ ---
        plate_found_this_frame = False
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            xcar1, ycar1, xcar2, ycar2, car_id = -1, -1, -1, -1, -1
            for track in track_ids:
                txc1, tyc1, txc2, tyc2, tid = track
                if x1 > txc1 and y1 > tyc1 and x2 < txc2 and y2 < tyc2:
                    xcar1, ycar1, xcar2, ycar2, car_id = txc1, tyc1, txc2, tyc2, int(tid)
                    break

            if car_id == -1:
                continue

            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
            if license_plate_crop.size == 0:
                continue

            # pokud auto uz ma zamknutou SPZ, nemen ji
            if car_id in car_locked:
                plate_found_this_frame = True
                last_plate_crop = license_plate_crop
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                continue

            plate_rect = make_plate_rectangle(license_plate_crop)
            new_readings = read_plate_5engines(plate_rect)

            if car_id not in car_readings:
                car_readings[car_id] = []
            car_readings[car_id].extend(new_readings)

            voted = vote_readings(car_readings[car_id])
            if voted:
                spz_formatted = voted[:3] + ' ' + voted[3:]
                car_best[car_id] = spz_formatted

                add_detection(spz_formatted, len(car_readings[car_id]), car_id, frame_nmr)

                # po 5+ cteni zamkni SPZ pro toto auto
                if len(car_readings[car_id]) >= 5:
                    car_locked[car_id] = True

            plate_found_this_frame = True
            last_plate_crop = license_plate_crop

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)

            if car_id in car_best:
                print(f'[FRAME {frame_nmr}] Car {car_id}: {car_best[car_id]} ({len(car_readings[car_id])}x)')

        # 5s timeout
        if plate_found_this_frame:
            last_plate_time = time.time()
        elif time.time() - last_plate_time > 5:
            car_readings.clear()
            car_best.clear()
            car_locked.clear()
            last_plate_crop = None

        # boxy kolem aut
        for track in track_ids:
            xc1, yc1, xc2, yc2, tid = track
            cv2.rectangle(frame, (int(xc1), int(yc1)), (int(xc2), int(yc2)), (0, 255, 0), 2)

        # aktualni SPZ seznam
        spz_counts = {}
        for cid, txt in car_best.items():
            n = len(car_readings[cid])
            if txt in spz_counts:
                spz_counts[txt] += n
            else:
                spz_counts[txt] = n
        sorted_spz = sorted(spz_counts.items(), key=lambda x: x[1], reverse=True)

        # nakresli SPZ na frame
        y_offset = 50
        for txt, count in sorted_spz:
            label = f'{txt}  ({count}x)'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)
            cv2.rectangle(frame, (10, y_offset - th - 8), (10 + tw + 10, y_offset + 8), (0, 0, 0), -1)
            cv2.putText(frame, label, (15, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 3)
            y_offset += th + 20

        # zvetseny vyrez SPZ v pravem dolnim rohu
        if last_plate_crop is not None and last_plate_crop.size > 0:
            fh, fw = frame.shape[:2]
            crop_w = 300
            crop_h = int(last_plate_crop.shape[0] * crop_w / max(last_plate_crop.shape[1], 1))
            if crop_h > 5:
                resized_plate = cv2.resize(last_plate_crop, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC)
                pad = 10
                rx = fw - crop_w - pad
                ry = fh - crop_h - pad
                cv2.rectangle(frame, (rx - 3, ry - 3), (rx + crop_w + 3, ry + crop_h + 3), (0, 0, 255), 2)
                frame[ry:ry + crop_h, rx:rx + crop_w] = resized_plate

        # zakoduj frame do JPEG a uloz do sdileneho stavu
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with lock:
            current_frame_jpg = jpeg.tobytes()
            current_spz_list = [{"text": txt, "count": cnt} for txt, cnt in sorted_spz]


# --- FLASK ENDPOINTY ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with lock:
                frame = current_frame_jpg
            if frame is None:
                blank = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(blank, 'Nacitam...', (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,100,100), 2)
                _, buf = cv2.imencode('.jpg', blank)
                frame = buf.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/current')
def api_current():
    with lock:
        spz = list(current_spz_list)
    return jsonify(spz)


@app.route('/api/history')
def api_history():
    with json_lock:
        data = load_json()

    # seskup podle plate_text, max count, posledni timestamp
    grouped = {}
    for item in data:
        txt = item["plate_text"]
        if txt not in grouped:
            grouped[txt] = {"text": txt, "timestamp": item["timestamp"], "count": item["count"]}
        else:
            if item["count"] > grouped[txt]["count"]:
                grouped[txt]["count"] = item["count"]
            if item["timestamp"] > grouped[txt]["timestamp"]:
                grouped[txt]["timestamp"] = item["timestamp"]

    # filtruj 5+ a serad od nejnovejsiho
    history = [v for v in grouped.values() if v["count"] >= 5]
    history.sort(key=lambda x: x["timestamp"], reverse=True)

    return jsonify(history[:200])


@app.route('/api/history/<plate_text>', methods=['DELETE'])
def api_delete(plate_text):
    with json_lock:
        data = load_json()
        data = [item for item in data if item["plate_text"] != plate_text]
        save_json(data)
    return jsonify({"ok": True})


if __name__ == '__main__':

    # spust detekci v pozadi
    det_thread = threading.Thread(target=detection_loop, daemon=True)
    det_thread.start()

    print('[WEB] Server bezi na http://127.0.0.1:8080')
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)