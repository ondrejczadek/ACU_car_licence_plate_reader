from ultralytics import YOLO
import cv2
import numpy as np
import json
import datetime
import time
import torch
import os
import logging
import warnings

# potlac warningy
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
logging.getLogger('ppocr').setLevel(logging.ERROR)
logging.getLogger('paddle').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

import pytesseract
import easyocr
from paddleocr import PaddleOCR
from PIL import Image
from collections import Counter

from sort.sort import *

results = {}
mot_tracker = Sort()

# --- CONFIG ---
TESS_CFG = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
source = './video_auta3.mp4'

# use Apple GPU if available
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'[INFO] Device: {device}')

# --- NACTI MODELY ---
print('[INFO] 1/6 YOLO vehicle...')
coco_model = YOLO('yolov8n.pt').to(device)

print('[INFO] 2/6 YOLO plate...')
license_plate_detector = YOLO('license_plate_detector.pt').to(device)

print('[INFO] 3/6 EasyOCR...')
easy_reader = easyocr.Reader(['en'], gpu=False, verbose=False)

print('[INFO] 4/6 PaddleOCR...')
paddle_reader = PaddleOCR(lang='en')

print('[INFO] 5/6 docTR...')
from doctr.models import recognition_predictor
doctr_reader = recognition_predictor(pretrained=True)

print('[INFO] 6/6 TrOCR...')
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')

print('[INFO] Vsechny modely nacteny.\n')

vehicles = [2, 3, 5, 7]

# --- POMOCNE FUNKCE ---

def clean_text(text):
    return ''.join(c for c in text.upper() if c.isalnum())


def order_pts(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def make_plate_rectangle(plate_crop):
    h, w = plate_crop.shape[:2]
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 30, 120)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kern, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_quad = None
    best_area = 0
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        area = cv2.contourArea(cnt)
        if area < 0.1 * w * h:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) == 4 and area > best_area:
            best_quad = approx.reshape(4, 2).astype(np.float32)
            best_area = area

    if best_quad is not None and best_area > 0.15 * w * h:
        src = order_pts(best_quad)
    else:
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ys, xs = np.where(binary > 0)
        if len(xs) < 10:
            return cv2.resize(plate_crop, (400, 80), interpolation=cv2.INTER_CUBIC)
        pts = np.column_stack((xs, ys)).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect).astype(np.float32)
        src = order_pts(box)

    w_s = np.linalg.norm(src[1] - src[0])
    h_s = np.linalg.norm(src[3] - src[0])
    if h_s > w_s:
        src = np.array([src[3], src[0], src[1], src[2]], dtype=np.float32)

    ctr = np.mean(src, axis=0)
    src = src + (src - ctr) * 0.05

    dst = np.array([[0, 0], [399, 0], [399, 79], [0, 79]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    result = cv2.warpPerspective(plate_crop, M, (400, 80),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
    return result


def remove_blue_strip(plate_img):
    hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([85, 30, 30]), np.array([140, 255, 255]))
    col_sums = np.sum(blue_mask > 0, axis=0)
    blue_cols = np.where(col_sums > plate_img.shape[0] * 0.15)[0]

    if len(blue_cols) > 0:
        cut = int(blue_cols[-1]) + 8
        if cut < plate_img.shape[1] * 0.3:
            return plate_img[:, cut:]

    min_cut = int(plate_img.shape[1] * 0.05)
    return plate_img[:, min_cut:]


# --- 5 OCR ENGINU ---

def read_tesseract(img):
    text = pytesseract.image_to_string(img, config=TESS_CFG).strip()
    return clean_text(text)

def read_easyocr(img):
    res = easy_reader.readtext(img, detail=0,
                                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    return clean_text(''.join(res))

def read_paddleocr(img):
    try:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        result = paddle_reader.predict(img)
        texts = []
        if result:
            for item in result:
                if hasattr(item, '__getitem__') and 'rec_texts' in item:
                    texts.extend(item['rec_texts'])
        return clean_text(''.join(texts))
    except Exception:
        return ''

def read_doctr(img):
    try:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        result = doctr_reader([img])
        texts = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    for word in line.words:
                        texts.append(word.value)
        return clean_text(''.join(texts))
    except Exception:
        return ''

def read_trocr(img):
    try:
        if len(img.shape) == 2:
            pil_img = Image.fromarray(img)
        else:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pixel_values = trocr_processor(images=pil_img, return_tensors="pt").pixel_values
        generated_ids = trocr_model.generate(pixel_values, max_new_tokens=20)
        text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return clean_text(text)
    except Exception:
        return ''


def read_plate_5engines(plate_img):
    """Precte SPZ vsemi 5 enginy, vrati seznam platnych cteni (7-8 znaku)."""
    plate_no_blue = remove_blue_strip(plate_img)

    # roztazeni na 400x80 obdelnik
    plate_rect = cv2.resize(plate_no_blue, (400, 80), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(plate_rect, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    readings = []
    for eng_func in [read_tesseract, read_easyocr, read_paddleocr, read_doctr, read_trocr]:
        text = eng_func(thresh)
        if len(text) == 7 or len(text) == 8:
            readings.append(text)

    return readings


def vote_readings(readings):
    if not readings:
        return None
    if len(readings) == 1:
        return readings[0]

    lengths = [len(r) for r in readings]
    target_len = Counter(lengths).most_common(1)[0][0]
    valid = [r for r in readings if len(r) == target_len]

    voted = ''
    for i in range(target_len):
        chars = [r[i] for r in valid]
        voted += Counter(chars).most_common(1)[0][0]

    if len(voted) == 7 or len(voted) == 8:
        return voted
    return None


# --- JSON DATABAZE ---
JSON_PATH = 'detections.json'

def load_json():
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, 'r') as f:
            return json.load(f)
    return []

def save_detection(plate_text, score, car_id, frame_number):
    data = load_json()
    now = datetime.datetime.now()
    merged = False
    for i in range(len(data) - 1, -1, -1):
        if data[i]["plate_text"] == plate_text:
            last_ts = datetime.datetime.fromisoformat(data[i]["timestamp"])
            if (now - last_ts).total_seconds() < 2:
                data[i]["count"] = max(data[i]["count"], score)
                data[i]["timestamp"] = now.isoformat()
                merged = True
            break
    if not merged:
        data.append({
            "timestamp": now.isoformat(),
            "plate_text": plate_text,
            "count": score,
            "car_id": car_id,
            "frame": frame_number
        })
    with open(JSON_PATH, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# --- HLAVNI SMYCKA ---
cap = cv2.VideoCapture(source)

# accumulated readings per car_id: {car_id: [list of valid readings]}
car_readings = {}
# best result per car_id: {car_id: "XXX XXXX"}
car_best = {}
car_locked = {}
last_plate_crop = None

frame_nmr = -1
ret = True
last_plate_time = time.time()

while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if not ret:
        break

    results[frame_nmr] = {}

    # --- DETEKCE AUT ---
    detections = coco_model(frame)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # --- TRACKING ---
    if len(detections_) > 0:
        track_ids = mot_tracker.update(np.asarray(detections_))
    else:
        track_ids = mot_tracker.update(np.empty((0, 5)))

    # --- DETEKCE SPZ ---
    plate_found_this_frame = False
    license_plates = license_plate_detector(frame)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # najdi ke kterymu autu patri
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

        # narovnani do obdelniku
        plate_rect = make_plate_rectangle(license_plate_crop)

        # precti 5 enginy
        new_readings = read_plate_5engines(plate_rect)

        # pridej ke kumulovanym ctenim pro toto auto
        if car_id not in car_readings:
            car_readings[car_id] = []
        car_readings[car_id].extend(new_readings)

        # hlasuj ze vsech dosavadnich cteni
        voted = vote_readings(car_readings[car_id])
        if voted:
            spz_formatted = voted[:3] + ' ' + voted[3:]
            car_best[car_id] = spz_formatted

            # uloz do results
            results[frame_nmr][car_id] = {
                'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                'license_plate': {
                    'bbox': [x1, y1, x2, y2],
                    'text': spz_formatted,
                    'bbox_score': score,
                    'text_score': len(car_readings[car_id])
                }
            }

            # uloz do JSON
            save_detection(spz_formatted, len(car_readings[car_id]), car_id, frame_nmr)

            # po 5+ cteni zamkni SPZ pro toto auto
            if len(car_readings[car_id]) >= 5:
                car_locked[car_id] = True

        plate_found_this_frame = True
        last_plate_crop = license_plate_crop

        # nakresli box kolem SPZ
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)

        if car_id in car_best:
            print(f'[FRAME {frame_nmr}] Car {car_id}: {car_best[car_id]} ({len(car_readings[car_id])}x)')

    # aktualizuj cas posledni detekce / vycisti seznam po 5s bez detekce
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

    # --- LEVY HORNI ROH: unikatni SPZ serazene podle vyskytu ---
    # secti stene SPZ dohromady (ruzna car_id mohou mit stejnou SPZ)
    spz_counts = {}
    for cid, txt in car_best.items():
        n = len(car_readings[cid])
        if txt in spz_counts:
            spz_counts[txt] += n
        else:
            spz_counts[txt] = n

    # serad od nejvysiho vyskytu
    sorted_spz = sorted(spz_counts.items(), key=lambda x: x[1], reverse=True)

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

    # live okno
    cv2.imshow('YOLO + SPZ (5 enginu)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print('\n[INFO] Hotovo.')
print(f'[INFO] Rozpoznane SPZ (serazene podle vyskytu):')
final_counts = {}
for cid, txt in car_best.items():
    n = len(car_readings[cid])
    if txt in final_counts:
        final_counts[txt] += n
    else:
        final_counts[txt] = n
for txt, count in sorted(final_counts.items(), key=lambda x: x[1], reverse=True):
    print(f'  {txt}  ({count}x)')