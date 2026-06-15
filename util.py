from ultralytics import YOLO
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
from paddleocr import PaddleOCR
from PIL import Image
from collections import Counter
from doctr.models import recognition_predictor
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# --- CONFIG ---
IMAGE_DIR = './dataset/images'
TESS_CFG = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
source = './auta_pokus_SPZ.mp4'
JSON_PATH = 'detections.json'
vehicles = [2, 3, 5, 7]

# load models

# use Apple GPU if available
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'[INFO] Device: {device}')

coco_model = YOLO('models/yolov8n.pt').to(device)
license_plate_detector = YOLO('models/license_plate_detector1.0.pt').to(device)
easy_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
paddle_reader = PaddleOCR(lang='en')
doctr_reader = recognition_predictor(pretrained=True)
trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')

def warnings():
    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
    logging.getLogger('ppocr').setLevel(logging.ERROR)
    logging.getLogger('paddle').setLevel(logging.ERROR)
    warnings.filterwarnings('ignore')

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

