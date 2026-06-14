import string
import cv2
import numpy as np
import pytesseract
from collections import Counter

# Tesseract config: single line, whitelist only plate characters
TESS_LINE = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
TESS_VERT = '--oem 3 --psm 5 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# Mapping dictionaries for character conversion (OCR error correction)
dict_char_to_int = {'O': '0', 'Q': '0', 'D': '0',
                    'I': '1', 'L': '1',
                    'J': '3',
                    'A': '4',
                    'S': '5',
                    'G': '6', 'b': '6',
                    'T': '7',
                    'B': '8'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '5': 'S',
                    '6': 'G',
                    '7': 'T',
                    '8': 'B'}


def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                car_id,
                                '[{} {} {} {}]'.format(
                                    results[frame_nmr][car_id]['car']['bbox'][0],
                                    results[frame_nmr][car_id]['car']['bbox'][1],
                                    results[frame_nmr][car_id]['car']['bbox'][2],
                                    results[frame_nmr][car_id]['car']['bbox'][3]),
                                '[{} {} {} {}]'.format(
                                    results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                    results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                    results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                    results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                results[frame_nmr][car_id]['license_plate']['text'],
                                results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()


def _order_points(pts):
    """Order 4 points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left = smallest x+y
    rect[2] = pts[np.argmax(s)]   # bottom-right = largest x+y
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]   # top-right = smallest y-x
    rect[3] = pts[np.argmax(d)]   # bottom-left = largest y-x
    return rect


def straighten_plate(plate_crop):
    """
    Perspektivni korekce SPZ: narovna do obdelniku 400x80,
    jako bychom se na ni koukali rovne zepredu.
    Orientaci urcuje podle modreho EU prouzku (vzdy vlevo).
    """
    h, w = plate_crop.shape[:2]

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- NAJDI 4 ROHY SPZ ---
    # canny edges
    edges = cv2.Canny(blur, 30, 120)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kern, iterations=2)

    # najdi kontury a hledej nejvetsi 4-uhelnik
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
        src_pts = _order_points(best_quad)
    else:
        # fallback: minAreaRect na vsech tmavych pixelech
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ys, xs = np.where(binary > 0)
        if len(xs) < 10:
            return cv2.resize(plate_crop, (400, 80), interpolation=cv2.INTER_CUBIC)

        points = np.column_stack((xs, ys)).astype(np.float32)
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect).astype(np.float32)
        src_pts = _order_points(box)

    # zajisti landscape (sirsi nez vyssi)
    w_side = np.linalg.norm(src_pts[1] - src_pts[0])
    h_side = np.linalg.norm(src_pts[3] - src_pts[0])
    if h_side > w_side:
        src_pts = np.array([src_pts[3], src_pts[0], src_pts[1], src_pts[2]], dtype=np.float32)

    # maly padding kolem bodu, aby se neorezaly znaky
    center = np.mean(src_pts, axis=0)
    src_pts = src_pts + (src_pts - center) * 0.05

    # perspective warp do rovneho obdelniku
    dst_pts = np.array([[0, 0], [399, 0], [399, 79], [0, 79]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    result = cv2.warpPerspective(plate_crop, M, (400, 80),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)

    # --- ORIENTACE: modry EU prouzek musi byt VLEVO ---
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, np.array([90, 40, 40]), np.array([135, 255, 255]))
    left_blue = np.sum(blue_mask[:, :80] > 0)    # levy sloupec
    right_blue = np.sum(blue_mask[:, 320:] > 0)  # pravy sloupec

    if right_blue > left_blue + 50:
        # modry prouzek je vpravo -> SPZ je vzhuru nohama
        result = cv2.rotate(result, cv2.ROTATE_180)

    return result


def filter_black_text(plate_img):
    """
    Filter only black text from the plate image.
    Keeps only dark characters, removes colors (EU strip, background).
    """
    hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)

    # black text: low value = dark pixels
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 100])
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # invert: black text on white background
    result = cv2.bitwise_not(mask)

    # clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    return result


def _ocr_single(img, config=TESS_LINE):
    """Run tesseract on one image, return cleaned text."""
    text = pytesseract.image_to_string(img, config=config).strip()
    return text.upper().replace(' ', '').replace('-', '').replace('.', '').replace('\n', '')


def _vote_characters(readings):
    """
    Vote on each character position across multiple OCR readings.
    Pick the most common character at each position.
    """
    if not readings:
        return ''

    # find the most common length
    lengths = [len(r) for r in readings]
    target_len = Counter(lengths).most_common(1)[0][0]

    # filter to readings with that length
    valid = [r for r in readings if len(r) == target_len]
    if not valid:
        return readings[0] if readings else ''

    # vote per position
    result = ''
    for i in range(target_len):
        chars = [r[i] for r in valid]
        most_common = Counter(chars).most_common(1)[0][0]
        result += most_common

    return result


def multi_scan_ocr(plate_img):
    """
    4-rotation OCR scan system.
    Reads the plate in 4 orientations and votes on the best result.
    """
    readings = []

    # scan 1: normal (0°)
    text_0 = _ocr_single(plate_img, TESS_LINE)
    if text_0:
        readings.append(text_0)

    # scan 2: rotated 90° clockwise
    rotated_90 = cv2.rotate(plate_img, cv2.ROTATE_90_CLOCKWISE)
    text_90 = _ocr_single(rotated_90, TESS_VERT)
    if text_90:
        readings.append(text_90)

    # scan 3: rotated 180° (upside down)
    rotated_180 = cv2.rotate(plate_img, cv2.ROTATE_180)
    text_180 = _ocr_single(rotated_180, TESS_LINE)
    if text_180:
        text_180 = text_180[::-1]
        readings.append(text_180)

    # scan 4: rotated 270° clockwise
    rotated_270 = cv2.rotate(plate_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    text_270 = _ocr_single(rotated_270, TESS_VERT)
    if text_270:
        text_270 = text_270[::-1]
        readings.append(text_270)

    print(f'  [4-scan] 0°="{text_0}" | 90°="{text_90}" | 180°="{text_180}" | 270°="{text_270}"')

    voted = _vote_characters(readings)
    return voted


def license_complies_format(text):
    if len(text) != 7:
        return False

    # format 1: D L D D D D D (standard czech)
    if (text[0] in ['0','1','2','3','4','5','6','7','8','9'] or text[0] in dict_char_to_int.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0','1','2','3','4','5','6','7','8','9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0','1','2','3','4','5','6','7','8','9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in ['0','1','2','3','4','5','6','7','8','9'] or text[4] in dict_char_to_int.keys()) and \
       (text[5] in ['0','1','2','3','4','5','6','7','8','9'] or text[5] in dict_char_to_int.keys()) and \
       (text[6] in ['0','1','2','3','4','5','6','7','8','9'] or text[6] in dict_char_to_int.keys()):
        return 'standard'

    # format 2: L L L D D D D (old czech)
    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in string.ascii_uppercase or text[2] in dict_int_to_char.keys()) and \
       (text[3] in ['0','1','2','3','4','5','6','7','8','9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in ['0','1','2','3','4','5','6','7','8','9'] or text[4] in dict_char_to_int.keys()) and \
       (text[5] in ['0','1','2','3','4','5','6','7','8','9'] or text[5] in dict_char_to_int.keys()) and \
       (text[6] in ['0','1','2','3','4','5','6','7','8','9'] or text[6] in dict_char_to_int.keys()):
        return 'old'

    return False


def format_license(text, fmt):
    license_plate_ = ''

    if fmt == 'standard':
        mapping = {0: dict_char_to_int, 1: dict_int_to_char,
                   2: dict_char_to_int, 3: dict_char_to_int, 4: dict_char_to_int,
                   5: dict_char_to_int, 6: dict_char_to_int}
        for j in [0, 1, 2, 3, 4, 5, 6]:
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]
        return license_plate_[:2] + license_plate_[2] + ' ' + license_plate_[3:]

    elif fmt == 'old':
        mapping = {0: dict_int_to_char, 1: dict_int_to_char, 2: dict_int_to_char,
                   3: dict_char_to_int, 4: dict_char_to_int, 5: dict_char_to_int,
                   6: dict_char_to_int}
        for j in [0, 1, 2, 3, 4, 5, 6]:
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]
        return license_plate_[:3] + ' ' + license_plate_[3:]

    return text


def read_license_plate(plate_img_color, plate_img_thresh):
    """
    Multi-scan OCR: 4 rotations, vote on result, then format.
    """
    filtered = filter_black_text(plate_img_color)

    voted_text = multi_scan_ocr(filtered)

    if voted_text:
        print(f'  [VOTED] "{voted_text}"')

        fmt = license_complies_format(voted_text)
        if fmt:
            return format_license(voted_text, fmt), 1.0

        if len(voted_text) >= 5:
            return voted_text, 0.5

    return None, None


def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1
