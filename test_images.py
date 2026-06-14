from ultralytics import YOLO
import cv2
import numpy as np
import torch
import os
import glob

from util import read_license_plate, straighten_plate

# use Apple GPU if available
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'[INFO] Device: {device}')

# load model
license_plate_detector = YOLO('license_plate_detector.pt').to(device)

# load test images
test_dir = './dataset/valid/images'
image_files = sorted(glob.glob(os.path.join(test_dir, '*.png')) +
                    glob.glob(os.path.join(test_dir, '*.jpg')))

print(f'[INFO] Testovani {len(image_files)} obrazku z {test_dir}')
print(f'[INFO] Kazdy obrazek 5 vterin, q = preskocit, ESC = ukoncit\n')

for idx, img_path in enumerate(image_files):
    img = cv2.imread(img_path)
    filename = os.path.basename(img_path)
    display = img.copy()
    plate_texts = []
    last_warped = None

    # detect license plates (snizeny threshold aby naslo vzdy)
    plates = license_plate_detector(img, conf=0.15)[0]

    for box in plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = box
        license_plate_crop = img[int(y1):int(y2), int(x1):int(x2), :]

        if license_plate_crop.size == 0:
            continue

        # --- NAROVNANI SPZ DO OBDELNIKU ---
        plate_warped = straighten_plate(license_plate_crop)
        last_warped = plate_warped

        # preprocessing pro OCR
        plate_gray = cv2.cvtColor(plate_warped, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        plate_gray = clahe.apply(plate_gray)
        plate_gray = cv2.bilateralFilter(plate_gray, 11, 17, 17)
        plate_thresh = cv2.adaptiveThreshold(
            plate_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # OCR
        license_plate_text, license_plate_text_score = read_license_plate(plate_warped, plate_thresh)

        if license_plate_text is not None:
            plate_texts.append(license_plate_text)

        # box kolem SPZ na hlavnim obrazku
        cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)

    # info v terminalu
    status = f'[{idx+1}/{len(image_files)}] {filename}'
    if plate_texts:
        status += f' -> {", ".join(plate_texts)}'
    else:
        status += ' -> zadna SPZ'
    print(status)

    # --- OCR TEXT V ROHU OBRAZKU ---
    ocr_display = ', '.join(plate_texts) if plate_texts else '???'
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(ocr_display, font, 1.5, 3)[0]
    pad = 10
    cv2.rectangle(display, (0, 0), (text_size[0] + 2*pad, text_size[1] + 2*pad + 10), (0, 0, 0), -1)
    cv2.putText(display, ocr_display, (pad, text_size[1] + pad), font, 1.5, (0, 255, 0), 3)

    # hlavni okno s fotkou
    cv2.imshow('Test SPZ', display)

    # vedlejsi okno s narovnanou SPZ (zvetsena)
    if last_warped is not None:
        warped_big = cv2.resize(last_warped, (600, 120), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Narovnana SPZ', warped_big)

    key = cv2.waitKey(5000) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()
print('\n[INFO] Test dokoncen.')