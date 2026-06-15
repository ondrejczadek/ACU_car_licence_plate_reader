import os
import glob

from util import *

suppress_warnings()

# load all images
image_files = sorted(
    glob.glob(os.path.join(IMAGE_DIR, '*.png')) +
    glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))
)
print(f'[INFO] {len(image_files)} obrazku z {IMAGE_DIR}')
print(f'[INFO] 5 OCR enginu: Tesseract + EasyOCR + PaddleOCR + docTR + TrOCR\n')

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
    results = license_plate_detector(img, conf=0.15)[0]

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