from ultralytics import YOLO
from paddleocr import PaddleOCR
from PIL import Image
from collections import Counter
from doctr.models import recognition_predictor
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import cv2
import os
import json
import numpy as np
import pytesseract
import datetime
import torch
import logging
import warnings
import easyocr
import threading


json_lock = threading.Lock()

# --- CONFIG ---
IMAGE_DIR = './dataset/images'
TESS_CFG = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
source = './auta_pokus_SPZ.mp4'
JSON_PATH = 'test_results/detections.json'
vehicles = [2, 3, 5, 7]

# use Apple GPU if available
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'[INFO] Device: {device}')

# load models
coco_model = YOLO('models/yolov8n.pt').to(device)
license_plate_detector = YOLO('models/license_plate_detector1.0.pt').to(device)
easy_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
paddle_reader = PaddleOCR(lang='en')
doctr_reader = recognition_predictor(pretrained=True)
trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')

# --- ignore warnings ---
def suppress_warnings():
    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
    logging.getLogger('ppocr').setLevel(logging.ERROR)
    logging.getLogger('paddle').setLevel(logging.ERROR)
    warnings.filterwarnings('ignore')

# --- PREPROCESSING function ---
def clean_text(text): # not use
    return ''.join(c for c in text.upper() if c.isalnum())

def order_pts(pts): # not use
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

def preprocess_variants(plate_color):
    """5 preprocessingu pro cerne znaky."""
    if len(plate_color.shape) == 3:
        gray = cv2.cvtColor(plate_color, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_color
    variants = []

    # 1) Otsu
    _, v1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(v1)

    # 2) CLAHE + adaptive
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    cl = clahe.apply(gray)
    cl = cv2.bilateralFilter(cl, 9, 75, 75)
    v2 = cv2.adaptiveThreshold(cl, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 15, 4)
    variants.append(v2)

    # 3) silny CLAHE + Otsu
    clahe2 = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    cl2 = clahe2.apply(gray)
    _, v3 = cv2.threshold(cl2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(v3)

    # 4) sharpen + Otsu
    sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(gray, -1, sharp)
    _, v4 = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(v4)

    # 5) gaussian + Otsu
    gb = cv2.GaussianBlur(gray, (3, 3), 0)
    _, v5 = cv2.threshold(gb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(v5)

    return variants

# --- OCR engines ---

def read_tesseract(img):
    text = pytesseract.image_to_string(img, config=TESS_CFG).strip()
    return clean_text(text)

def read_easyocr(img):
    res = easy_reader.readtext(img, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
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

# --- OCR function ---

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

# --- JSON operations ---

def load_json():
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, 'r') as f:
            return json.load(f)
    return []

def save_json(data):
    with open(JSON_PATH, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def add_detection(plate_text, score, car_id, frame_number):
    with json_lock:
        data = load_json()
        now = datetime.datetime.now()
        # pokud posledni zaznam se stejnou SPZ je < 2s stary, aktualizuj ho
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
        save_json(data)

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