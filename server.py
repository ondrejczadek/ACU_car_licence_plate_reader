from util import *
from sort.sort import Sort
from flask import Flask, Response, jsonify, render_template, request
import time
import threading

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

    mot_tracker = Sort()
    cap = cv2.VideoCapture(source)
    car_readings = {}
    car_best = {}
    car_locked = {}
    last_plate_crop = None
    last_plate_time = time.time()
    is_running = True

    while True:
        frame_nmr += 1
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_nmr = -1
            car_readings.clear()
            car_best.clear()
            mot_tracker = Sort()
            continue

        frame, car_readings, car_best, car_locked, last_plate_crop, last_plate_time, sorted_spz = \
            process_frame(frame, mot_tracker, car_readings, car_best, car_locked, last_plate_crop, last_plate_time)

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
    det_thread = threading.Thread(target=detection_loop, daemon=True)
    det_thread.start()

    print('[WEB] Server bezi na http://127.0.0.1:8080')
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)