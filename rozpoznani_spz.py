from ultralytics import YOLO
import cv2
import numpy as np
import torch
import os
import glob
import logging
import warnings
import pytesseract
import easyocr
from paddleocr import PaddleOCR
from collections import Counter
from PIL import Image

# potlac warningy
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
logging.getLogger('ppocr').setLevel(logging.ERROR)
logging.getLogger('paddle').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

# --- CONFIG ---
IMAGE_DIR = './dataset/images'
TESS_CFG = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# use Apple GPU if available
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'[INFO] Device: {device}')

# --- NACTI MODELY ---
print('[INFO] 1/5 Nacitam YOLO...')
plate_detector = YOLO('license_plate_detector.pt').to(device)

print('[INFO] 2/5 Nacitam EasyOCR...')
easy_reader = easyocr.Reader(['en'], gpu=False, verbose=False)

print('[INFO] 3/5 Nacitam PaddleOCR...')
paddle_reader = PaddleOCR(lang='en')

print('[INFO] 4/5 Nacitam docTR...')
from doctr.models import recognition_predictor
doctr_reader = recognition_predictor(pretrained=True)

print('[INFO] 5/5 Nacitam TrOCR...')
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')

# load all images
image_files = sorted(
    glob.glob(os.path.join(IMAGE_DIR, '*.png')) +
    glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))
)
print(f'[INFO] {len(image_files)} obrazku z {IMAGE_DIR}')
print(f'[INFO] 5 OCR enginu: Tesseract + EasyOCR + PaddleOCR + docTR + TrOCR\n')


# --- POMOCNE FUNKCE ---

def clean_text(text):
    """Pouze velka pismena a cislice."""
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
    """Narovnani SPZ do obdelniku 400x80."""
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
    """Agresivne orizne modry EU prouzek."""
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


# --- 5 OCR ENGINU ---

def read_tesseract(img):
    text = pytesseract.image_to_string(img, config=TESS_CFG).strip()
    return clean_text(text)


def read_easyocr(img):
    results = easy_reader.readtext(img, detail=0,
                                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    return clean_text(''.join(results))


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
        # doctr ocekava numpy array (H, W, 3)
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


def vote_readings(readings):
    """Hlasovani po znacich - prevaha o 1 staci."""
    if not readings:
        return ''
    if len(readings) == 1:
        return readings[0]

    lengths = [len(r) for r in readings]
    target_len = Counter(lengths).most_common(1)[0][0]
    valid = [r for r in readings if len(r) == target_len]

    voted = ''
    for i in range(target_len):
        chars = [r[i] for r in valid]
        voted += Counter(chars).most_common(1)[0][0]
    return voted


def try_read_plate(plate_img):
    """
    5 OCR enginu x 5 preprocessingu = 25 cteni.
    Filtruje jen 7-8 znaku (ceska SPZ).
    Hlasuje po znacich, prevaha o 1 staci.
    """
    plate_no_blue = remove_blue_strip(plate_img)
    variants = preprocess_variants(plate_no_blue)
    all_readings = []

    engines = [
        ('Tess  ', read_tesseract),
        ('Easy  ', read_easyocr),
        ('Paddle', read_paddleocr),
        ('docTR ', read_doctr),
        ('TrOCR ', read_trocr),
    ]

    for eng_name, eng_func in engines:
        for i, processed in enumerate(variants):
            text = eng_func(processed)
            if len(text) == 7 or len(text) == 8:
                all_readings.append(text)
                print(f'    [{eng_name} {i+1}] "{text}" OK')
            else:
                print(f'    [{eng_name} {i+1}] "{text}" ({len(text)}z)')

    if not all_readings:
        return None

    if len(all_readings) == 1:
        result = all_readings[0]
        print(f'    >>> "{result}" (1 platny scan)')
        return result[:3] + ' ' + result[3:]

    # hlasovani ze vsech enginu
    voted = vote_readings(all_readings)
    if len(voted) != 7 and len(voted) != 8:
        return None

    match_count = sum(1 for r in all_readings if r == voted)
    total = len(all_readings)
    print(f'    >>> "{voted}" (shoda {match_count}/{total})')
    return voted[:3] + ' ' + voted[3:]


# --- HLAVNI SMYCKA ---
print('[INFO] ENTER = dalsi, ESC = konec\n')

for idx, img_path in enumerate(image_files):
    img = cv2.imread(img_path)
    if img is None:
        continue
    filename = os.path.basename(img_path)
    display = img.copy()

    # detekce SPZ
    results = plate_detector(img, conf=0.15)[0]

    if len(results.boxes) == 0:
        print(f'[{idx+1}/{len(image_files)}] {filename} -> ZADNA SPZ')
        continue

    boxes = results.boxes.data.tolist()
    boxes.sort(key=lambda b: b[4], reverse=True)
    x1, y1, x2, y2, score, class_id = boxes[0]

    plate_crop = img[int(y1):int(y2), int(x1):int(x2), :]
    if plate_crop.size == 0:
        continue

    plate_rect = make_plate_rectangle(plate_crop)

    print(f'[{idx+1}/{len(image_files)}] {filename}:')
    spz_text = try_read_plate(plate_rect)

    if spz_text is None:
        spz_text = '???'
    print(f'  ==> SPZ: {spz_text}\n')

    # zobrazeni
    cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(spz_text, font, 1.5, 3)[0]
    pad = 10
    cv2.rectangle(display, (0, 0),
                  (text_size[0] + 2*pad, text_size[1] + 2*pad + 10),
                  (0, 0, 0), -1)
    cv2.putText(display, spz_text, (pad, text_size[1] + pad),
                font, 1.5, (0, 255, 0), 3)
    cv2.imshow('Auto + SPZ', display)

    plate_no_blue = remove_blue_strip(plate_rect)
    plate_big = cv2.resize(plate_no_blue, (600, 120), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('SPZ vyrez (bez modre)', plate_big)

    key = cv2.waitKey(0) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()
print('\n[INFO] Hotovo.')