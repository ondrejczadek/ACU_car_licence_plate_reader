  # Bezpecnostni audit: License Plate Camera

  **Projekt:** License Plate Camera - Detektor registracnich znacek  
  **Verze auditu:** 1.0  
  **Datum:** 2026-04-04  
  **Metodologie:** OWASP Top 10 (2021), OWASP ASVS 4.0.3  
  **Klasifikace:** INTERNÍ - DŮVĚRNÉ

  ---

  ## 1. Manažerské shrnutí

  Projekt **License Plate Camera** je systém pro detekci a rozpoznávání registračních značek vozidel v reálném čase. Využívá YOLO modely pro detekci vozidel a SPZ, 5 OCR enginů (Tesseract, EasyOCR, PaddleOCR, docTR, TrOCR) s hlasovacím mechanismem a Flask webový server pro vizualizaci výsledků.

  ### Celkové hodnocení bezpečnostní úrovně: KRITICKÉ

  Systém vykazuje **24 bezpečnostních nálezů**, z toho:

  | Závažnost | Počet |
  |-----------|-------|
  | Kritická  | 6     |
  | Vysoká    | 8     |
  | Střední   | 6     |
  | Nízká     | 4     |

  **Klíčový závěr:** Systém v současném stavu **není vhodný pro produkční nasazení**. Zpracovává osobní údaje (registrační značky vozidel podléhající GDPR) bez jakékoliv autentizace, autorizace, šifrování či auditního logování.

  ---

  ## 2. Architektura systému

  ### 2.1 Komponenty

  ```
  ┌─────────────────────────────────────────────────────────┐
  │                    Flask Web Server                       │
  │                  (0.0.0.0:8080, HTTP)                    │
  │                                                           │
  │  Endpointy:                                              │
  │    GET  /              → HTML UI (index.html)            │
  │    GET  /video_feed    → MJPEG stream (real-time)        │
  │    GET  /api/current   → JSON aktuální detekce           │
  │    GET  /api/history   → JSON historie detekcí           │
  │    DELETE /api/history/<plate> → smazání záznamu          │
  │                                                           │
  │  ┌──────────────────────────────────────────────────┐    │
  │  │         Detekční vlákno (daemon thread)           │    │
  │  │                                                    │    │
  │  │  Video vstup → YOLO Vehicle → YOLO Plate →        │    │
  │  │  → SORT Tracker → 5x OCR → Hlasování →            │    │
  │  │  → detections.json                                 │    │
  │  └──────────────────────────────────────────────────┘    │
  └─────────────────────────────────────────────────────────┘
                            │
                            ▼
                ┌──────────────────────┐
                │   detections.json    │
                │  (plaintext, PII)    │
                └──────────────────────┘
  ```

  ### 2.2 Datový tok

  1. Video vstup (lokální soubor `.mp4`) → OpenCV `VideoCapture`
  2. Detekce vozidel → YOLO v8 (COCO model, třídy 2/3/5/7)
  3. Tracking vozidel → SORT algoritmus (Kalman filter)
  4. Detekce SPZ → YOLO (vlastní model `license_plate_detector.pt`)
  5. OCR → 5 enginů paralelně, hlasování po znacích
  6. Uložení → `detections.json` (plaintext JSON)
  7. Prezentace → Flask HTTP endpoints, MJPEG stream

  ### 2.3 Technologický stack

  | Komponenta | Technologie | Verze |
  |------------|-------------|-------|
  | Web server | Flask | neuvedena v requirements |
  | Detekce objektů | YOLOv8 (ultralytics) | 8.0.114 |
  | OCR #1 | Tesseract (pytesseract) | neuvedena |
  | OCR #2 | EasyOCR | 1.7.0 |
  | OCR #3 | PaddleOCR | neuvedena |
  | OCR #4 | docTR | neuvedena |
  | OCR #5 | TrOCR (HuggingFace) | neuvedena |
  | Tracking | SORT + filterpy | 1.4.5 |
  | ML framework | PyTorch | neuvedena |
  | Zpracování obrazu | OpenCV | 4.7.0.72 |

  ---

  ## 3. OWASP Top 10 (2021) - Mapování nálezů

  ### A01:2021 - Broken Access Control

  **Stav: KRITICKÝ - Kompletní absence řízení přístupu**

  #### Nález SEC-001: Žádná autentizace na žádném endpointu
  - **Závažnost:** Kritická
  - **CVSS 3.1:** 9.8 (Critical)
  - **Umístění:** `server.py:428-488` (všechny Flask routes)
  - **Popis:** Všech 5 endpointů aplikace je přístupných bez jakékoliv autentizace. Kdokoliv na síti může:
    - Sledovat živý video stream (`/video_feed`)
    - Číst aktuální detekce (`/api/current`)
    - Číst kompletní historii registračních značek (`/api/history`)
    - Mazat libovolné záznamy (`/api/history/<plate>`)
  - **Důkaz:**
    ```python
    # server.py:458-479
    @app.route('/api/history')
    def api_history():
        # ŽÁDNÁ autentizační kontrola
        with json_lock:
            data = load_json()
        # ... vrací kompletní historii detekcí
    ```
  - **Dopad:** Neoprávněný přístup k osobním údajům (SPZ), neoprávněné mazání důkazů, sledování pohybu vozidel.
  - **ASVS:** V2.1.1, V4.1.1, V4.1.2, V4.1.3

  #### Nález SEC-002: Žádná autorizace (BOLA) na DELETE endpointu
  - **Závažnost:** Kritická
  - **CVSS 3.1:** 9.1 (Critical)
  - **Umístění:** `server.py:482-488`
  - **Popis:** Endpoint pro mazání záznamů neprovádí žádnou kontrolu oprávnění. Neexistuje koncept vlastnictví dat ani rolí.
  - **Důkaz:**
    ```python
    @app.route('/api/history/<plate_text>', methods=['DELETE'])
    def api_delete(plate_text):
        with json_lock:
            data = load_json()
            data = [item for item in data if item["plate_text"] != plate_text]
            save_json(data)
        return jsonify({"ok": True})  # Vždy vrátí ok, i pro neexistující záznamy
    ```
  - **Dopad:** Kdokoliv může smazat libovolný záznam SPZ. Zničení důkazního materiálu. Žádný audit trail mazání.
  - **ASVS:** V4.2.1, V4.2.2

  #### Nález SEC-003: Chybějící CSRF ochrana
  - **Závažnost:** Vysoká
  - **CVSS 3.1:** 8.1 (High)
  - **Umístění:** `server.py:482-488`, `templates/index.html:159-162`
  - **Popis:** DELETE operace jsou volány prostým `fetch()` bez CSRF tokenů. Útočník může přimět přihlášeného uživatele k nechtěnému smazání záznamů.
  - **Důkaz:**
    ```javascript
    // index.html:159-161
    async function smazat(text) {
        await fetch('/api/history/' + encodeURIComponent(text), {method: 'DELETE'});
        // Žádný CSRF token
    }
    ```
  - **ASVS:** V4.2.2

  #### Nález SEC-004: Server naslouchá na všech síťových rozhraních
  - **Závažnost:** Vysoká
  - **CVSS 3.1:** 7.5 (High)
  - **Umístění:** `server.py:498`
  - **Popis:** Flask server je vázán na `0.0.0.0`, což znamená, že naslouchá na všech síťových rozhraních, včetně veřejných.
  - **Důkaz:**
    ```python
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
    ```
  - **Dopad:** Server je dostupný komukoliv ve stejné síti. V kombinaci s absencí autentizace je celý systém otevřený.
  - **ASVS:** V9.1.1

  ---

  ### A02:2021 - Cryptographic Failures

  **Stav: KRITICKÝ - Žádné šifrování**

  #### Nález SEC-005: Žádné šifrování dat v klidu (data at rest)
  - **Závažnost:** Kritická
  - **CVSS 3.1:** 7.5 (High)
  - **Umístění:** `server.py:125-133`, soubor `detections.json`
  - **Popis:** Registrační značky (osobní údaje dle GDPR) jsou ukládány v plaintext JSON souboru bez jakéhokoliv šifrování.
  - **Důkaz:**
    ```python
    def save_json(data):
        with open(JSON_PATH, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    ```
    ```json
    {
      "timestamp": "2026-04-03T21:42:48.735899",
      "plate_text": "5J5 7859",
      "count": 1,
      "car_id": 83,
      "frame": 1218
    }
    ```
  - **Dopad:** Každý, kdo získá přístup k souborovému systému, může přečíst kompletní historii detekovaných SPZ.
  - **ASVS:** V6.1.1, V6.2.1

  #### Nález SEC-006: Žádné šifrování při přenosu (data in transit)
  - **Závažnost:** Kritická
  - **CVSS 3.1:** 7.4 (High)
  - **Umístění:** `server.py:498`
  - **Popis:** Veškerá komunikace probíhá přes nešifrovaný HTTP protokol. Video stream, JSON data se SPZ, i DELETE požadavky jsou přenášeny v plaintextu.
  - **Dopad:** Man-in-the-middle útoky, odposlech síťové komunikace, zachycení registračních značek.
  - **ASVS:** V9.1.1, V9.1.2, V9.1.3

  ---

  ### A03:2021 - Injection

  **Stav: VYSOKÝ**

  #### Nález SEC-007: Cross-Site Scripting (XSS) ve webovém rozhraní
  - **Závažnost:** Vysoká
  - **CVSS 3.1:** 6.1 (Medium)
  - **Umístění:** `templates/index.html:145-155`
  - **Popis:** Data z API (`d.text`, `d.count`) jsou vkládána přímo do HTML pomocí řetězcové konkatenace bez escapování. Pokud OCR engine vrátí text obsahující speciální HTML znaky, může dojít k XSS.
  - **Důkaz:**
    ```javascript
    // index.html:148-154
    return '<tr>' +
        '<td class="col-spz">' + d.text + '</td>' +     // BEZ ESCAPOVÁNÍ
        '<td class="col-akce"><button class="smazat" onclick="smazat(\'' + d.text + '\')">' +
        // Injection přes onclick handler
    ```
  - **Vektor útoku:** Pokud by útočník dokázal manipulovat s obsahem `detections.json` nebo pokud by OCR engine interpretoval zkreslený obraz jako text obsahující `<script>` či uvozovky, XSS je možný.
  - **ASVS:** V5.3.3, V5.3.4

  #### Nález SEC-008: Nedostatečná validace vstupu na DELETE endpointu
  - **Závažnost:** Střední
  - **CVSS 3.1:** 5.3 (Medium)
  - **Umístění:** `server.py:482-488`
  - **Popis:** Parametr `plate_text` v URL není nijak validován na formát české SPZ. Přijímá se jakýkoliv řetězec.
  - **Důkaz:**
    ```python
    @app.route('/api/history/<plate_text>', methods=['DELETE'])
    def api_delete(plate_text):
        # Žádná validace formátu plate_text
        data = [item for item in data if item["plate_text"] != plate_text]
    ```
  - **ASVS:** V5.1.1, V5.1.3

  ---

  ### A04:2021 - Insecure Design

  **Stav: VYSOKÝ**

  #### Nález SEC-009: Chybějící bezpečnostní architektura (Security by Design)
  - **Závažnost:** Vysoká
  - **CVSS 3.1:** N/A (designový problém)
  - **Popis:** Systém byl navržen bez jakéhokoliv bezpečnostního designu:
    - Žádný threat model
    - Žádné bezpečnostní požadavky
    - Žádný princip least privilege
    - Žádná vrstva defence-in-depth
    - Zpracování PII dat (SPZ) bez ochranných mechanismů
  - **ASVS:** V1.1.1, V1.1.2, V1.1.3, V1.1.4, V1.1.5, V1.1.6, V1.1.7

  #### Nález SEC-010: Chybějící oddělení vrstev (Separation of Concerns)
  - **Závažnost:** Střední
  - **Umístění:** `server.py` (498 řádků, mixuje web server + ML pipeline + data access)
  - **Popis:** Webový server, ML inference pipeline a datová vrstva jsou sloučeny v jednom souboru bez oddělení. Detekční vlákno běží jako daemon thread se sdílenými globálními proměnnými.
  - **Dopad:** Zranitelnost v jedné vrstvě kompromituje celý systém. Obtížné bezpečnostní testování.
  - **ASVS:** V1.1.2

  ---

  ### A05:2021 - Security Misconfiguration

  **Stav: VYSOKÝ**

  #### Nález SEC-011: Hardcoded konfigurace a cesty
  - **Závažnost:** Střední
  - **CVSS 3.1:** 5.3 (Medium)
  - **Umístění:** `server.py:32-33`, `main.py:31`, `dataset/data.yaml`
  - **Důkaz:**
    ```python
    # server.py
    source = './auta_pokus_SPZ.mp4'
    JSON_PATH = 'detections.json'

    # dataset/data.yaml
    train: /Users/vaclavstraser/Desktop/License_plate_camera-main/dataset/train/images
    ```
  - **Dopad:** Absolutní cesty prozrazují uživatelské jméno a strukturu souborového systému. Konfigurace vyžaduje změny ve zdrojovém kódu.
  - **ASVS:** V14.2.1

  #### Nález SEC-012: Chybějící rate limiting
  - **Závažnost:** Vysoká
  - **CVSS 3.1:** 7.5 (High)
  - **Umístění:** Všechny Flask endpointy v `server.py`
  - **Popis:** Žádný endpoint nemá omezení počtu požadavků. Video stream (`/video_feed`) generuje neomezený MJPEG tok.
  - **Dopad:** DoS útoky, vyčerpání zdrojů serveru, masivní extrakce dat.
  - **ASVS:** V13.2.2

  #### Nález SEC-013: Chybějící bezpečnostní HTTP hlavičky
  - **Závažnost:** Střední
  - **Umístění:** `server.py` (Flask app)
  - **Chybějící hlavičky:**
    - `Content-Security-Policy`
    - `X-Frame-Options`
    - `X-Content-Type-Options`
    - `Strict-Transport-Security`
    - `Referrer-Policy`
    - `Permissions-Policy`
  - **ASVS:** V14.4.1, V14.4.2, V14.4.3, V14.4.4, V14.4.5, V14.4.6, V14.4.7

  #### Nález SEC-014: Potlačení bezpečnostních varování
  - **Závažnost:** Nízká
  - **Umístění:** `server.py:15-18`, `main.py:13-16`
  - **Důkaz:**
    ```python
    logging.getLogger('ppocr').setLevel(logging.ERROR)
    logging.getLogger('paddle').setLevel(logging.ERROR)
    warnings.filterwarnings('ignore')  # Potlačí VŠECHNA varování
    ```
  - **Dopad:** Skrytí bezpečnostních varování z knihoven. Bezpečnostní upozornění PaddleOCR nebo PyTorch jsou ignorována.
  - **ASVS:** V7.1.1

  ---

  ### A06:2021 - Vulnerable and Outdated Components

  **Stav: STŘEDNÍ**

  #### Nález SEC-015: Neúplné a zastaralé závislosti
  - **Závažnost:** Střední
  - **CVSS 3.1:** 5.3 (Medium)
  - **Umístění:** `requirements.txt`
  - **Popis:**
    - Uvedeno pouze 7 ze ~15 skutečných závislostí
    - Chybí: Flask, torch, pytesseract, paddleocr, doctr, transformers, Pillow
    - Verze jsou z roku 2023, potenciálně obsahují známé CVE
    - Žádná hash verifikace balíčků (`--hash` flag v pip)
    - Žádný `requirements.lock` pro reprodukovatelnost
  - **Důkaz:**
    ```
    ultralytics==8.0.114    # z června 2023
    opencv-python==4.7.0.72 # z dubna 2023
    numpy==1.24.3           # z dubna 2023
    ```
  - **ASVS:** V14.2.1, V14.2.2

  #### Nález SEC-016: Neověřené stahování ML modelů
  - **Závažnost:** Střední
  - **Umístění:** `server.py:195-196`, `main.py:56-57`
  - **Důkaz:**
    ```python
    trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
    trocr_model_ocr = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
    ```
  - **Popis:** ML modely jsou stahovány z HuggingFace Hub bez ověření integrity (hash/checksum). Útočník ovládající síť může podvrhnout škodlivý model.
  - **ASVS:** V14.2.4

  ---

  ### A07:2021 - Identification and Authentication Failures

  **Stav: KRITICKÝ - Kompletní absence**

  #### Nález SEC-017: Žádný autentizační mechanismus
  - **Závažnost:** Kritická
  - **Umístění:** Celá aplikace
  - **Popis:** Systém nemá:
    - Žádné uživatelské účty
    - Žádné přihlašování
    - Žádné API klíče
    - Žádné tokeny
    - Žádné sessions
    - Žádný `SECRET_KEY` pro Flask
  - **ASVS:** V2.1.1, V2.2.1, V3.1.1, V3.2.1, V3.3.1, V3.4.1

  ---

  ### A08:2021 - Software and Data Integrity Failures

  **Stav: STŘEDNÍ**

  #### Nález SEC-018: Neověřená integrita datového souboru
  - **Závažnost:** Střední
  - **Umístění:** `server.py:125-133`
  - **Popis:** Soubor `detections.json` nemá žádný mechanismus pro ověření integrity (checksum, digitální podpis). Modifikace souboru zůstane nezjištěna.
  - **ASVS:** V10.3.1

  #### Nález SEC-019: Neatomické operace se souborem
  - **Závažnost:** Střední
  - **Umístění:** `server.py:131-133`, `main.py:266-267`
  - **Důkaz:**
    ```python
    # server.py - s json_lock
    def save_json(data):
        with open(JSON_PATH, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # main.py - BEZ jakéhokoliv locku
    def save_detection(plate_text, score, car_id, frame_number):
        # ...
        with open(JSON_PATH, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    ```
  - **Popis:** Zápis do JSON souboru není atomický. Při pádu systému během zápisu dojde ke ztrátě/poškození dat. V `main.py` navíc chybí thread-safe ochrana.
  - **ASVS:** V8.1.1

  ---

  ### A09:2021 - Security Logging and Monitoring Failures

  **Stav: KRITICKÝ**

  #### Nález SEC-020: Žádné bezpečnostní logování
  - **Závažnost:** Kritická
  - **Umístění:** Celá aplikace
  - **Popis:** Systém nemá:
    - Žádné logování přístupů k API
    - Žádné logování DELETE operací (smazání záznamů)
    - Žádné logování neúspěšných požadavků
    - Žádný audit trail
    - Žádné logování do zabezpečeného externího úložiště
    - Žádné alertování při podezřelé aktivitě
  - **Existující logování:**
    ```python
    # Pouze stdout printy pro debugging:
    print(f'[FRAME {frame_nmr}] Car {car_id}: {car_best[car_id]}')
    print(f'[ERROR] Detection crashed: {e}')
    ```
  - **Dopad:** Nelze detekovat útoky, nelze provést forenzní analýzu, nelze splnit povinnost notifikace dle GDPR (72 hodin).
  - **ASVS:** V7.1.1, V7.1.2, V7.2.1, V7.2.2, V7.3.1, V7.3.2, V7.4.1

  ---

  ### A10:2021 - Server-Side Request Forgery (SSRF)

  **Stav: NÍZKÝ**

  #### Nález SEC-021: Hardcoded video zdroj omezuje riziko SSRF
  - **Závažnost:** Nízká
  - **Umístění:** `server.py:32`
  - **Popis:** Video zdroj je hardcoded jako lokální soubor (`./auta_pokus_SPZ.mp4`). Pokud by se v budoucnu umožnilo zadávání URL jako zdroje videa (např. RTSP stream), vzniklo by riziko SSRF přes `cv2.VideoCapture()`.
  - **Stav:** Potenciální riziko při rozšíření funkcionality.
  - **ASVS:** V12.3.1

  ---

  ## 4. Dodatečné nálezy mimo OWASP Top 10

  #### Nález SEC-022: Únik citlivých informací přes chybové výpisy
  - **Závažnost:** Nízká
  - **Umístění:** `server.py:167-169`
  - **Důkaz:**
    ```python
    except Exception as e:
        print(f'[ERROR] Detection crashed: {e}')
        import traceback
        traceback.print_exc()  # Kompletní stack trace na stdout
    ```
  - **Dopad:** Únik informací o interní struktuře kódu, cestách k souborům, verzích knihoven.

  #### Nález SEC-023: Neomezený MJPEG stream bez timeoutu
  - **Závažnost:** Střední
  - **Umístění:** `server.py:435-448`
  - **Důkaz:**
    ```python
    def generate():
        while True:  # Nekonečná smyčka bez podmínky ukončení
            # ...
            yield (b'--frame\r\n' ...)
            time.sleep(0.03)  # ~33 FPS
    ```
  - **Dopad:** Každé otevřené spojení na `/video_feed` spotřebovává server resources neomezeně. Útočník může otevřít mnoho spojení a vyčerpat zdroje.

  #### Nález SEC-024: Wildcard import z SORT modulu
  - **Závažnost:** Nízká
  - **Umístění:** `main.py:24`
  - **Důkaz:**
    ```python
    from sort.sort import *  # Importuje VŠE včetně potenciálně nebezpečných symbolů
    ```
  - **Dopad:** Namespace pollution, potenciální přepsání built-in funkcí.

  ---

  ## 5. OWASP ASVS 4.0.3 - Detailní hodnocení

  ### Úroveň ověření

  | ASVS Úroveň | Popis | Stav projektu |
  |-------------|-------|--------------|
  | Level 1 | Základní (oportunistické hrozby) | **NESPLNĚNO** |
  | Level 2 | Standardní (citlivá data) | NESPLNĚNO |
  | Level 3 | Pokročilé (kritická data) | NESPLNĚNO |

  **Doporučená cílová úroveň:** ASVS Level 2 (systém zpracovává osobní údaje - SPZ)

  ### V1: Architektura, Design a Threat Modeling

  | Požadavek | ID | Stav | Poznámka |
  |-----------|-----|------|----------|
  | Threat model existuje | V1.1.1 | NESPLNĚNO | Žádný threat model |
  | Bezpečnostní kontroly identifikovány | V1.1.2 | NESPLNĚNO | Žádné bezpečnostní kontroly |
  | Komponenty odděleny | V1.1.3 | ČÁSTEČNĚ | Web + ML v jednom procesu |
  | High-level architektura definována | V1.1.4 | NESPLNĚNO | Žádná dokumentace |
  | Kryptografie centralizována | V1.1.5 | NESPLNĚNO | Žádná kryptografie |
  | Princip least privilege | V1.1.6 | NESPLNĚNO | Vše běží pod jedním uživatelem |

  ### V2: Autentizace

  | Požadavek | ID | Stav | Poznámka |
  |-----------|-----|------|----------|
  | Silná autentizace | V2.1.1 | NESPLNĚNO | Žádná autentizace |
  | Anti-automatizace | V2.2.1 | NESPLNĚNO | Žádná ochrana proti botům |
  | Credential recovery | V2.5.1 | N/A | Žádné credentials |
  | Lookup secret | V2.6.1 | NESPLNĚNO | Žádné API klíče |

  ### V3: Session Management

  | Požadavek | ID | Stav | Poznámka |
  |-----------|-----|------|----------|
  | Session tokeny | V3.1.1 | NESPLNĚNO | Žádné sessions |
  | Session timeout | V3.3.1 | NESPLNĚNO | - |
  | Cookie security flags | V3.4.1 | NESPLNĚNO | Žádné cookies |

  ### V4: Access Control

  | Požadavek | ID | Stav | Poznámka |
  |-----------|-----|------|----------|
  | Princip nejmenších oprávnění | V4.1.1 | NESPLNĚNO | Vše veřejné |
  | RBAC nebo ABAC | V4.1.2 | NESPLNĚNO | Žádné role |
  | Ochrana citlivých dat | V4.1.3 | NESPLNĚNO | SPZ volně dostupné |
  | Object-level access control | V4.2.1 | NESPLNĚNO | Žádný BOLA check |
  | CSRF ochrana | V4.2.2 | NESPLNĚNO | Žádné CSRF tokeny |

  ### V5: Validace, Sanitizace a Kódování

  | Požadavek | ID | Stav | Poznámka |
  |-----------|-----|------|----------|
  | Input validace na serveru | V5.1.1 | NESPLNĚNO | Žádná validace plate_text |
  | Structured data validace | V5.1.3 | NESPLNĚNO | - |
  | Output encoding pro HTML | V5.3.3 | NESPLNĚNO | XSS v index.html |
  | Context-aware output encoding | V5.3.4 | NESPLNĚNO | innerHTML bez escapování |

  ### V6: Kryptografie v klidu

  | Požadavek | ID | Stav | Poznámka |
  |-----------|-----|------|----------|
  | Šifrování citlivých dat | V6.1.1 | NESPLNĚNO | Plaintext JSON |
  | Správa šifrovacích klíčů | V6.2.1 | NESPLNĚNO | Žádné klíče |
  | Použití schválených algoritmů | V6.2.2 | N/A | Žádná kryptografie |

  ### V7: Logování a Monitoring

  | Požadavek | ID | Stav | Poznámka |
  |-----------|-----|------|----------|
  | Logování bezpečnostních událostí | V7.1.1 | NESPLNĚNO | Pouze stdout |
  | Logování autentizace | V7.1.2 | NESPLNĚNO | - |
  | Ochrana integrity logů | V7.2.1 | NESPLNĚNO | - |
  | Logování přístupu k citlivým datům | V7.2.2 | NESPLNĚNO | - |
  | Detekce anomálií | V7.3.1 | NESPLNĚNO | - |

  ### V8: Ochrana dat

  | Požadavek | ID | Stav | Poznámka |
  |-----------|-----|------|----------|
  | Ochrana citlivých dat v paměti | V8.1.1 | NESPLNĚNO | Globální proměnné |
  | Data minimization | V8.1.2 | ČÁSTEČNĚ | Ukládá se car_id, frame# |

  ### V9: Komunikace

  | Požadavek | ID | Stav | Poznámka |
  |-----------|-----|------|----------|
  | TLS pro všechnu komunikaci | V9.1.1 | NESPLNĚNO | Pouze HTTP |
  | Ověření TLS certifikátů | V9.1.2 | NESPLNĚNO | - |
  | TLS 1.2+ | V9.1.3 | NESPLNĚNO | - |

  ### V13: API a webové služby

  | Požadavek | ID | Stav | Poznámka |
  |-----------|-----|------|----------|
  | Ověření všech API zdrojů dat | V13.1.1 | NESPLNĚNO | - |
  | Rate limiting | V13.2.2 | NESPLNĚNO | - |

  ### V14: Konfigurace

  | Požadavek | ID | Stav | Poznámka |
  |-----------|-----|------|----------|
  | Bezpečná default konfigurace | V14.2.1 | NESPLNĚNO | Hardcoded hodnoty |
  | Aktuální závislosti | V14.2.2 | NESPLNĚNO | Verze z 2023 |
  | Security headers | V14.4.1-7 | NESPLNĚNO | Žádné security headers |

  ---

  ## 6. Posouzení souladu s GDPR

  Registrační značky vozidel jsou **osobními údaji** dle čl. 4 odst. 1 GDPR (umožňují identifikaci fyzické osoby - vlastníka vozidla).

  | GDPR Požadavek | Článek | Stav | Problém |
  |----------------|--------|------|---------|
  | Právní základ zpracování | Čl. 6 | NEŘEŠENO | Není definován právní základ |
  | Data minimization | Čl. 5(1)(c) | ČÁSTEČNĚ | Ukládá se i car_id, frame |
  | Účelové omezení | Čl. 5(1)(b) | NEŘEŠENO | Není definován účel zpracování |
  | Omezení uložení | Čl. 5(1)(e) | NESPLNĚNO | Žádná data retention policy |
  | Integrita a důvěrnost | Čl. 5(1)(f) | NESPLNĚNO | Žádné šifrování, žádná autentizace |
  | DPIA (posouzení vlivu) | Čl. 35 | NEPROVEDENO | Vyžadováno pro systematické monitorování |
  | Transparentnost | Čl. 13/14 | NESPLNĚNO | Žádné informační cedule pro řidiče |
  | Právo na výmaz | Čl. 17 | ČÁSTEČNĚ | DELETE endpoint existuje, ale bez autentizace |
  | Notifikace incidentu | Čl. 33/34 | NEMOŽNÉ | Bez logování nelze detekovat incident |

  ---

  ## 7. Threat Model (STRIDE)

  ### Identifikované hrozby

  | Kategorie | Hrozba | Pravděpodobnost | Dopad | Riziko |
  |-----------|--------|-----------------|-------|--------|
  | **Spoofing** | Falešné požadavky na DELETE API | Vysoká | Vysoký | KRITICKÉ |
  | **Tampering** | Modifikace detections.json | Střední | Vysoký | VYSOKÉ |
  | **Repudiation** | Žádný audit trail = popření akcí | Vysoká | Vysoký | KRITICKÉ |
  | **Info Disclosure** | Čtení SPZ dat přes HTTP | Vysoká | Vysoký | KRITICKÉ |
  | **Info Disclosure** | Sledování video streamu | Vysoká | Vysoký | KRITICKÉ |
  | **Denial of Service** | Zahlcení /video_feed spojení | Střední | Střední | STŘEDNÍ |
  | **Elevation of Privilege** | N/A - žádné role | N/A | N/A | N/A |

  ---

  ## 8. Plán nápravy (Remediation Roadmap)

  ### Fáze 1 - KRITICKÉ (okamžitá implementace)

  #### 1.1 Implementace autentizace
  ```python
  # Doporučený přístup: API klíče + JWT pro webové rozhraní
  from functools import wraps
  import os

  def require_api_key(f):
      @wraps(f)
      def decorated(*args, **kwargs):
          key = request.headers.get('X-API-Key')
          if not key or key != os.environ.get('SPZ_API_KEY'):
              return jsonify({'error': 'Unauthorized'}), 401
          return f(*args, **kwargs)
      return decorated

  # Aplikovat na všechny endpointy:
  @app.route('/api/history')
  @require_api_key
  def api_history():
      ...
  ```

  #### 1.2 Přechod na HTTPS
  ```python
  # Varianta 1: Flask s SSL kontextem
  app.run(host='127.0.0.1', port=443,
          ssl_context=('cert.pem', 'key.pem'))

  # Varianta 2 (doporučená): Reverse proxy (nginx)
  # nginx.conf:
  # server {
  #     listen 443 ssl;
  #     ssl_certificate /path/to/cert.pem;
  #     ssl_certificate_key /path/to/key.pem;
  #     location / { proxy_pass http://127.0.0.1:8080; }
  # }
  ```

  #### 1.3 Omezení síťového rozhraní
  ```python
  # Změna v server.py:498
  app.run(host='127.0.0.1', port=8080, debug=False, threaded=True)
  #       ^^^^^^^^^^^^ místo 0.0.0.0
  ```

  #### 1.4 Šifrování dat v klidu
  ```python
  from cryptography.fernet import Fernet

  ENCRYPTION_KEY = os.environ.get('SPZ_ENCRYPTION_KEY')
  cipher = Fernet(ENCRYPTION_KEY.encode())

  def save_json_encrypted(data):
      plaintext = json.dumps(data).encode()
      encrypted = cipher.encrypt(plaintext)
      with open(JSON_PATH, 'wb') as f:
          f.write(encrypted)

  def load_json_encrypted():
      if not os.path.exists(JSON_PATH):
          return []
      with open(JSON_PATH, 'rb') as f:
          encrypted = f.read()
      plaintext = cipher.decrypt(encrypted)
      return json.loads(plaintext.decode())
  ```

  #### 1.5 Oprava XSS
  ```javascript
  // Nahradit innerHTML bezpečnou variantou:
  function escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
  }

  // V updateHistory():
  return '<tr>' +
      '<td class="col-spz">' + escapeHtml(d.text) + '</td>' +
      '<td class="col-akce"><button class="smazat" data-plate="' +
      escapeHtml(d.text) + '">&times;</button></td>' +
      '</tr>';

  // Event delegation místo inline onclick:
  document.getElementById('history').addEventListener('click', e => {
      if (e.target.classList.contains('smazat')) {
          smazat(e.target.dataset.plate);
      }
  });
  ```

  ### Fáze 2 - VYSOKÉ (1-2 týdny)

  #### 2.1 Bezpečnostní logování
  ```python
  import logging
  from datetime import datetime

  security_logger = logging.getLogger('security')
  handler = logging.FileHandler('/var/log/spz_detector/security.log')
  handler.setFormatter(logging.Formatter(
      '%(asctime)s [%(levelname)s] %(message)s'
  ))
  security_logger.addHandler(handler)
  security_logger.setLevel(logging.INFO)

  # Příklady logování:
  security_logger.info(f'API_ACCESS src={request.remote_addr} endpoint=/api/history')
  security_logger.warning(f'DELETE_RECORD src={request.remote_addr} plate={plate_text}')
  security_logger.warning(f'AUTH_FAIL src={request.remote_addr} endpoint={request.path}')
  ```

  #### 2.2 Rate limiting
  ```python
  from flask_limiter import Limiter
  from flask_limiter.util import get_remote_address

  limiter = Limiter(
      get_remote_address,
      app=app,
      default_limits=["100 per minute"],
      storage_uri="memory://"
  )

  @app.route('/api/history')
  @limiter.limit("30 per minute")
  def api_history():
      ...

  @app.route('/api/history/<plate_text>', methods=['DELETE'])
  @limiter.limit("10 per minute")
  def api_delete(plate_text):
      ...
  ```

  #### 2.3 CSRF ochrana
  ```python
  from flask_wtf.csrf import CSRFProtect

  app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY')
  csrf = CSRFProtect(app)

  # Pro API endpointy s API klíčem:
  @csrf.exempt
  @app.route('/api/history', methods=['GET'])
  def api_history():
      ...
  ```

  #### 2.4 Validace vstupu
  ```python
  import re

  CZECH_PLATE_REGEX = re.compile(r'^[0-9A-Z]{1,3}\s[0-9A-Z]{4,5}$')

  @app.route('/api/history/<plate_text>', methods=['DELETE'])
  def api_delete(plate_text):
      if not CZECH_PLATE_REGEX.match(plate_text):
          return jsonify({'error': 'Invalid plate format'}), 400
      ...
  ```

  #### 2.5 Bezpečnostní HTTP hlavičky
  ```python
  @app.after_request
  def set_security_headers(response):
      response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self'; script-src 'self'"
      response.headers['X-Frame-Options'] = 'DENY'
      response.headers['X-Content-Type-Options'] = 'nosniff'
      response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
      response.headers['Permissions-Policy'] = 'camera=(), microphone=()'
      return response
  ```

  ### Fáze 3 - STŘEDNÍ (3-4 týdny)

  | Úkol | Popis |
  |------|-------|
  | Konfigurace přes env proměnné | Přesunout všechny hardcoded hodnoty do `.env` |
  | Data retention policy | Implementovat automatické mazání záznamů starších N dní |
  | Atomické operace se soubory | Použít `tempfile` + `os.rename()` pro atomický zápis |
  | Aktualizace závislostí | Aktualizovat na nejnovější verze, přidat hash verifikaci |
  | Kompletní requirements.txt | Doplnit všechny skutečné závislosti |
  | Integritní kontroly ML modelů | Ověřovat SHA-256 hash stažených modelů |
  | Error handling | Implementovat globální error handler bez úniku informací |

  ### Fáze 4 - NÍZKÉ (vylepšení)

  | Úkol | Popis |
  |------|-------|
  | Docker kontejnerizace | Izolace aplikace v read-only kontejneru |
  | Penetrační testování | Profesionální bezpečnostní audit |
  | DPIA (Data Protection Impact Assessment) | Povinné dle GDPR čl. 35 |
  | Backup & recovery | Zálohování šifrovaných dat |
  | CI/CD bezpečnostní skenování | Integrace Bandit, Safety, Snyk |

  ---

  ## 9. Kontrolní seznam pro vývojáře

  ### Před nasazením do produkce:

  - [ ] Autentizace implementována na všech endpointech
  - [ ] HTTPS/TLS povoleno (TLS 1.2+)
  - [ ] Server vázán na localhost (127.0.0.1) nebo za reverse proxy
  - [ ] Data šifrována v klidu (detections.json)
  - [ ] XSS opraveno (output encoding v HTML šabloně)
  - [ ] CSRF ochrana aktivní
  - [ ] Validace vstupu na DELETE endpointu
  - [ ] Rate limiting nastaveno
  - [ ] Bezpečnostní HTTP hlavičky přidány
  - [ ] Security logging implementováno
  - [ ] Audit trail pro všechny CRUD operace
  - [ ] requirements.txt kompletní s hash verifikací
  - [ ] Závislosti aktualizovány na nejnovější bezpečné verze
  - [ ] Varování knihoven nejsou globálně potlačena
  - [ ] SECRET_KEY nakonfigurován přes environment variable
  - [ ] Hardcoded cesty nahrazeny konfigurovatelným nastavením
  - [ ] DPIA provedena (povinnost dle GDPR)
  - [ ] Informační cedule o kamerovém systému (zákonný požadavek)
  - [ ] Data retention policy definována a implementována

  ---

  ## 10. Závěr

  Projekt **License Plate Camera** demonstruje funkční detekci a rozpoznávání registračních značek s pokročilým multi-engine OCR přístupem. Z bezpečnostního hlediska však systém **nesplňuje ani základní úroveň (Level 1) standardu OWASP ASVS** a není v souladu s požadavky GDPR na zpracování osobních údajů.

  **Nejkritičtější nálezy:**
  1. Kompletní absence autentizace a autorizace
  2. Žádné šifrování dat (v klidu ani při přenosu)
  3. XSS zranitelnost v uživatelském rozhraní
  4. Žádné bezpečnostní logování
  5. Nesoulad s GDPR při zpracování registračních značek

  Implementace nápravných opatření Fáze 1 je nezbytná před jakýmkoliv testovacím nasazením mimo izolované vývojové prostředí.

  ---

  *Dokument byl vytvořen na základě statické analýzy zdrojového kódu. Pro kompletní bezpečnostní posouzení se doporučuje provést dynamické testování (DAST) a penetrační test.*
