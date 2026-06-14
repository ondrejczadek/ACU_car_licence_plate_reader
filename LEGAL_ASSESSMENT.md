# Právní a licenční posouzení: License Plate Camera

**Projekt:** License Plate Camera - Detektor registračních značek  
**Verze posouzení:** 1.0  
**Datum:** 2026-04-04  
**Klasifikace:** INTERNÍ - DŮVĚRNÉ

---

## 1. Manažerské shrnutí

Projekt License Plate Camera obsahuje **kritický licenční konflikt**, který činí současnou distribuci projektu **právně neplatnou**. Projekt je licencován pod MIT licencí, ale zahrnuje komponentu SORT pod GPLv3 a závisí na knihovně Ultralytics pod AGPL-3.0. Obě tyto copyleft licence vyžadují, aby celý odvozený projekt byl distribuován pod stejnou copyleft licencí. MIT licence tuto podmínku nesplňuje.

Dále systém zpracovává registrační značky vozidel, které jsou dle GDPR osobními údaji, bez splnění zákonných požadavků na ochranu osobních údajů. Provozování systému v České republice bez řádného právního rámce představuje porušení nařízení (EU) 2016/679 (GDPR) a zákona č. 110/2019 Sb.

### Celkové hodnocení

| Oblast | Stav | Závažnost |
|--------|------|-----------|
| Licenční kompatibilita | **PORUŠENÍ** - MIT vs. GPLv3/AGPL-3.0 konflikt | Kritická |
| Copyleft povinnosti | **NESPLNĚNY** - chybí zdrojový kód, AGPL network clause | Kritická |
| GDPR soulad | **NESPLNĚN** - zpracování PII bez právního základu | Kritická |
| Atribuce třetích stran | **NEÚPLNÁ** - chybí NOTICE soubor | Vysoká |
| ML modely - licence | **NEOVĚŘENA** - licence tréninkových dat nezjištěna | Střední |
| Copyright | Částečně v pořádku | Nízká |

---

## 2. Licence projektu

### 2.1 Deklarovaná licence

**Soubor:** `LICENSE`

```
MIT License

Copyright (c) 2026 Václav Dominik Štraser
Copyright (c) 2026 Antonín Majer
Copyright (c) 2026 Ondřej Czadek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

MIT licence je permisivní licence, která umožňuje:
- Volné použití, kopírování, modifikaci
- Distribuci a sublicencování
- Komerční i nekomerční použití

**Jediné povinnosti:** zachování copyright notice a licence v kopiích.

### 2.2 Problém: MIT licence je nekompatibilní se zahrnutými komponentami

Projekt **nemůže** být platně distribuován pod MIT licencí, protože obsahuje a závisí na komponentách s přísnějšími copyleft licencemi. Viz sekce 3 a 4.

---

## 3. Analýza licencí třetích stran

### 3.1 Přímé závislosti - kompletní přehled

| Komponenta | Licence | Typ | Kompatibilní s MIT? | Copyleft efekt |
|------------|---------|-----|---------------------|----------------|
| **SORT** (sort/) | **GPL-3.0** | Silný copyleft | **NE** | Celý projekt musí být GPL-3.0 |
| **Ultralytics** (YOLOv8) | **AGPL-3.0** | Silný copyleft + síťový | **NE** | Celý projekt musí být AGPL-3.0 |
| PyTorch | BSD-3-Clause | Permisivní | Ano | Žádný |
| OpenCV (cv2) | Apache-2.0 / MIT | Permisivní | Ano | Žádný |
| NumPy | BSD-3-Clause | Permisivní | Ano | Žádný |
| SciPy | BSD-3-Clause | Permisivní | Ano | Žádný |
| Pandas | BSD-3-Clause | Permisivní | Ano | Žádný |
| Flask | BSD-3-Clause | Permisivní | Ano | Žádný |
| EasyOCR | Apache-2.0 | Permisivní | Ano | Žádný |
| PaddleOCR | Apache-2.0 | Permisivní | Ano | Žádný |
| PaddlePaddle | Apache-2.0 | Permisivní | Ano | Žádný |
| docTR | Apache-2.0 | Permisivní | Ano | Žádný |
| Transformers (HF) | Apache-2.0 | Permisivní | Ano | Žádný |
| pytesseract | Apache-2.0 | Permisivní | Ano | Žádný |
| Pillow | HPND (PIL) | Permisivní | Ano | Žádný |
| filterpy | MIT | Permisivní | Ano | Žádný |
| lap | BSD-2-Clause | Permisivní | Ano | Žádný |
| scikit-image | BSD-3-Clause | Permisivní | Ano | Žádný |

### 3.2 Nepřímé / systémové závislosti

| Komponenta | Licence | Poznámka |
|------------|---------|----------|
| Tesseract OCR (systémový binární) | Apache-2.0 | Volán přes pytesseract wrapper |
| COCO dataset (YOLOv8 pre-trained) | CC BY 4.0 | Licence tréninkových dat |
| TrOCR model (microsoft/trocr-base-printed) | MIT | HuggingFace model card |

---

## 4. Kritické licenční konflikty

### 4.1 KONFLIKT #1: SORT modul (GPLv3) vs. MIT projekt

**Závažnost: KRITICKÁ**

**Fakta:**
- Složka `sort/` obsahuje SORT tracker pod **GNU General Public License v3** (Alex Bewley, Copyright 2016-2020)
- SORT kód je přímo zahrnut v repozitáři (nikoliv jako pip závislost, ale jako zkopírovaný zdrojový kód)
- Hlavní soubory projektu (`server.py:26`, `main.py:24`) importují SORT:
  ```python
  from sort.sort import Sort   # server.py
  from sort.sort import *       # main.py
  ```
- SORT kód je integrální součástí projektu - bez něj nefunguje vehicle tracking

**Právní důsledek dle GPL-3.0 § 5(c):**

> *"You must license the entire work, as a whole, under this License to anyone who comes into possession of a copy."*

Protože projekt **zahrnuje** GPL-3.0 kód a **vytváří s ním kombinované dílo** (combined work), celý projekt musí být licencován pod GPL-3.0 nebo kompatibilní licencí. MIT licence tuto podmínku nesplňuje.

**Stav: PORUŠENÍ GPL-3.0**

### 4.2 KONFLIKT #2: Ultralytics/YOLOv8 (AGPL-3.0) vs. MIT projekt

**Závažnost: KRITICKÁ**

**Fakta:**
- Knihovna `ultralytics` (YOLOv8) je licencována pod **GNU Affero General Public License v3.0**
- Projekt ji používá jako klíčovou závislost:
  ```python
  from ultralytics import YOLO                    # server.py:2, main.py:1
  coco_model = YOLO('yolov8n.pt').to(device)     # detekce vozidel
  license_plate_detector = YOLO('license_plate_detector.pt').to(device)  # detekce SPZ
  ```
- AGPL-3.0 je ještě přísnější než GPL-3.0 kvůli **network interaction clause** (§ 13)

**Právní důsledek dle AGPL-3.0 § 13:**

> *"If you modify the Program, your modified version must prominently offer all users interacting with it remotely through a computer network [...] an opportunity to receive the Corresponding Source of your version."*

Protože server.py provozuje Flask webový server, který **interaguje s uživateli přes počítačovou síť**, a zároveň používá AGPL-3.0 knihovnu Ultralytics, platí:

1. **Celý projekt** musí být licencován pod AGPL-3.0
2. **Zdrojový kód musí být zpřístupněn** všem uživatelům, kteří interagují se serverem přes síť
3. Webové rozhraní musí obsahovat odkaz na zdrojový kód

**Stav: PORUŠENÍ AGPL-3.0**

### 4.3 Vztah mezi GPL-3.0 a AGPL-3.0

GPL-3.0 (SORT) a AGPL-3.0 (Ultralytics) jsou vzájemně kompatibilní dle GPL-3.0 § 13 a AGPL-3.0 § 13. Výsledné dílo musí být licencováno pod **AGPL-3.0** (přísnější z obou), čímž jsou splněny požadavky obou licencí.

### 4.4 Dopad na projekt

| Aspekt | Současný stav | Požadovaný stav |
|--------|---------------|-----------------|
| Licence projektu | MIT | **AGPL-3.0** |
| Zdrojový kód | Není povinně zpřístupněn | **Musí být dostupný všem síťovým uživatelům** |
| Komerční uzavřený kód | Povoleno (MIT) | **Zakázáno** (bez komerční licence od Ultralytics) |
| Sublicencování | Povoleno (MIT) | **Zakázáno** (§ 10 AGPL) |
| Deriváty | Jakákoliv licence | **Musí být AGPL-3.0** |

---

## 5. Analýza atribučních povinností

### 5.1 Chybějící atribuce

Většina permisivních licencí vyžaduje zachování copyright notice a licence. Projekt v současnosti **nesplňuje** tyto atribuční povinnosti:

| Komponenta | Licence | Požadavek | Splněno? |
|------------|---------|-----------|----------|
| SORT | GPL-3.0 | Copyright notice + plný text GPL v distribuci | Ano (sort/LICENSE existuje) |
| Ultralytics | AGPL-3.0 | Copyright notice + plný text AGPL v distribuci | **NE** (jen pip závislost) |
| OpenCV | Apache-2.0 | Copyright notice + text licence + NOTICE soubor | **NE** |
| EasyOCR | Apache-2.0 | Copyright notice + text licence | **NE** |
| PaddleOCR | Apache-2.0 | Copyright notice + text licence | **NE** |
| PyTorch | BSD-3-Clause | Copyright notice + text licence | **NE** |
| TrOCR model | MIT | Copyright notice (Microsoft) | **NE** |
| COCO dataset | CC BY 4.0 | Atribuce autora datasetu | **NE** |

### 5.2 Chybějící NOTICE / THIRD_PARTY_LICENSES soubor

Projekt postrádá soubor s přehledem licencí třetích stran. Dle Apache-2.0 § 4(d) je distribuce NOTICE souboru povinná, pokud existuje.

### 5.3 Citační povinnosti

SORT README.md požaduje akademickou citaci při použití v research kontextu:

```bibtex
@inproceedings{Bewley2016_sort,
  author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
  title={Simple online and realtime tracking},
  year={2016},
  doi={10.1109/ICIP.2016.7533003}
}
```

YOLOv8/Ultralytics rovněž vyžaduje citaci. V README.md projektu ani jinde **žádné citace uvedeny nejsou**.

---

## 6. Analýza ML modelů

### 6.1 YOLOv8n.pt (pre-trained COCO model)

| Aspekt | Detail |
|--------|--------|
| Poskytovatel | Ultralytics |
| Licence modelu | AGPL-3.0 (součást Ultralytics distribuce) |
| Tréninková data | MS COCO dataset (CC BY 4.0) |
| Atribuční povinnost | Ano - COCO dataset vyžaduje atribuci |
| Komerční použití | Vyžaduje komerční licenci od Ultralytics |

### 6.2 license_plate_detector.pt (vlastní model)

| Aspekt | Detail |
|--------|--------|
| Poskytovatel | Vlastní (trénováno v projektu) |
| Framework | Ultralytics YOLOv8 (AGPL-3.0) |
| Tréninková data | Vlastní dataset (`dataset/`) |
| Licence modelu | **Nespecifikována** |
| Problém | Model trénovaný pomocí AGPL kódu - distribuce modelu podléhá AGPL? |

**Důležitá poznámka:** Právní otázka, zda AGPL-3.0 licence tréninkového frameworku (Ultralytics) se vztahuje i na výstupní model (váhy sítě), je **právně nevyjasněná**. Konzervativní výklad by zahrnoval model pod AGPL. Ultralytics nabízí komerční licenci, která toto řeší.

### 6.3 TrOCR model (microsoft/trocr-base-printed)

| Aspekt | Detail |
|--------|--------|
| Poskytovatel | Microsoft |
| Licence | MIT |
| Stahování | Za běhu z HuggingFace Hub |
| Komerční použití | Povoleno |
| Atribuce | Vyžadována (MIT) |

### 6.4 EasyOCR modely

| Aspekt | Detail |
|--------|--------|
| Poskytovatel | JaidedAI |
| Licence | Apache-2.0 |
| Stahování | Za běhu |
| Komerční použití | Povoleno |

### 6.5 PaddleOCR modely

| Aspekt | Detail |
|--------|--------|
| Poskytovatel | Baidu |
| Licence modelu | Apache-2.0 |
| Komerční použití | Povoleno |

### 6.6 docTR modely

| Aspekt | Detail |
|--------|--------|
| Poskytovatel | Mindee |
| Licence | Apache-2.0 |
| Komerční použití | Povoleno |

---

## 7. Soulad s GDPR a českým právem

### 7.1 Klasifikace zpracovávaných dat

Registrační značky (SPZ) vozidel jsou **osobními údaji** dle čl. 4 odst. 1 nařízení (EU) 2016/679 (GDPR):

> *"Osobními údaji [se rozumí] veškeré informace o identifikované či identifikovatelné fyzické osobě."*

SPZ umožňuje identifikaci vlastníka vozidla prostřednictvím registru vozidel. Tento výklad potvrzuje i stanovisko WP29 (nyní EDPB) 4/2007 a rozhodovací praxe ÚOOÚ.

### 7.2 Další zpracovávané osobní údaje

| Údaj | Kategorie | Uchovávání |
|------|-----------|------------|
| Registrační značka (plate_text) | Osobní údaj | detections.json (bez omezení) |
| Timestamp detekce | Osobní údaj (v kombinaci se SPZ) | detections.json |
| Video stream (snímky vozidel) | Osobní údaj (obraz vozidla) | V paměti (real-time) |
| Car tracking ID | Pseudonymní identifikátor | detections.json |
| Frame number | Technický údaj | detections.json |

### 7.3 Právní základ zpracování (čl. 6 GDPR)

**Stav: NEDEFINOVÁN**

Projekt nedefinuje právní základ pro zpracování osobních údajů. Možné právní základy:

| Právní základ | Článek | Použitelnost |
|---------------|--------|-------------|
| Souhlas subjektu údajů | Čl. 6(1)(a) | **Nepraktické** - nelze získat souhlas od řidičů projíždějících kolem kamery |
| Plnění smlouvy | Čl. 6(1)(b) | **Nepoužitelné** - žádná smlouva s řidiči |
| Právní povinnost | Čl. 6(1)(c) | Pouze pokud zákon ukládá monitorování (např. bezpečnostní předpisy) |
| Životně důležité zájmy | Čl. 6(1)(d) | **Nepoužitelné** |
| Veřejný zájem | Čl. 6(1)(e) | Pouze pro orgány veřejné moci (policie, obce) |
| **Oprávněný zájem** | **Čl. 6(1)(f)** | **Nejpravděpodobnější** - vyžaduje balanční test |

### 7.4 Požadavky GDPR - analýza souladu

| Požadavek | Článek GDPR | Stav | Popis problému |
|-----------|-------------|------|----------------|
| Právní základ | Čl. 6 | **NESPLNĚN** | Není definován žádný právní základ |
| Informační povinnost | Čl. 13/14 | **NESPLNĚNA** | Žádné informační cedule, žádná privacy policy |
| Záznamy o zpracování | Čl. 30 | **NESPLNĚN** | Neexistují záznamy o činnostech zpracování |
| DPIA | Čl. 35 | **NEPROVEDENA** | Systematické monitorování veřejného prostoru vyžaduje DPIA |
| Zabezpečení | Čl. 32 | **NESPLNĚN** | Žádné šifrování, žádná autentizace (viz Security Audit) |
| Minimalizace dat | Čl. 5(1)(c) | **ČÁSTEČNĚ** | Ukládá se car_id a frame number (nadbytečné) |
| Omezení uložení | Čl. 5(1)(e) | **NESPLNĚN** | Žádná data retention policy, data se hromadí neomezeně |
| Právo na přístup | Čl. 15 | **NESPLNĚN** | Žádný mechanismus pro subjekt údajů |
| Právo na výmaz | Čl. 17 | **ČÁSTEČNĚ** | DELETE endpoint existuje, ale bez autentizace (kdokoliv může mazat) |
| Právo na námitku | Čl. 21 | **NESPLNĚN** | Žádný mechanismus |
| Právo na přenositelnost | Čl. 20 | **NESPLNĚN** | Žádný mechanismus |
| Oznámení porušení | Čl. 33/34 | **NEMOŽNÉ** | Bez logování nelze detekovat incident |
| Pověřenec (DPO) | Čl. 37 | **NEURČENO** | Závisí na rozsahu a povaze zpracování |

### 7.5 DPIA - povinnost provedení

Dle čl. 35 GDPR je posouzení vlivu na ochranu osobních údajů (DPIA) **povinné**, pokud zpracování pravděpodobně představuje vysoké riziko pro práva a svobody fyzických osob. Dle Seznamu ÚOOÚ (Úřad pro ochranu osobních údajů) operací vyžadujících DPIA se jedná o:

- **Systematické monitorování veřejně přístupných prostorů** (kritérium WP248 č. 3)
- **Zpracování ve velkém rozsahu** (kritérium WP248 č. 5) - závisí na nasazení
- **Inovativní využití technologie** (kritérium WP248 č. 8) - AI/ML rozpoznávání

**Stav: DPIA NEBYLA PROVEDENA** - toto je porušení čl. 35 GDPR.

### 7.6 Český zákon č. 110/2019 Sb. (zákon o zpracování osobních údajů)

Zákon adaptuje GDPR do českého práva a stanoví:
- Pokuta za porušení: až 10 000 000 EUR nebo 2 % ročního obratu (čl. 83(4) GDPR), resp. až 20 000 000 EUR nebo 4 % ročního obrotu za závažná porušení (čl. 83(5) GDPR)
- ÚOOÚ je dozorový orgán

### 7.7 Zákon č. 449/2001 Sb. o myslivosti / Zákon č. 361/2000 Sb. o provozu na pozemních komunikacích

Kamerový systém snímající registrační značky na pozemních komunikacích může spadat pod specifické předpisy o provozu na pozemních komunikacích. Je třeba ověřit, zda:
- Umístění kamery nevyžaduje povolení správce komunikace
- Systém neporušuje předpisy o užívání pozemních komunikací

---

## 8. Regulace kamerových systémů v ČR

### 8.1 Obecné požadavky

Provozování kamerového systému se záznamem registračních značek v České republice podléhá:

1. **GDPR + zákon č. 110/2019 Sb.** - ochrana osobních údajů
2. **Zákon č. 89/2012 Sb. (občanský zákoník)** - § 86 (ochrana soukromí) a § 88 (zachycení podoby)
3. **Zákon č. 40/2009 Sb. (trestní zákoník)** - § 182 (porušení tajemství dopravovaných zpráv), § 183 (porušení tajemství listin)
4. **Stanoviska ÚOOÚ** - metodické pokyny k provozování kamerových systémů

### 8.2 Povinnosti provozovatele kamerového systému

| Povinnost | Právní základ | Stav v projektu |
|-----------|---------------|-----------------|
| Označení monitorovaného prostoru informační tabulkou | GDPR čl. 13, stanovisko ÚOOÚ | **NESPLNĚNO** |
| Stanovení účelu zpracování | GDPR čl. 5(1)(b) | **NESPLNĚNO** |
| Stanovení doby uchovávání záznamů | GDPR čl. 5(1)(e) | **NESPLNĚNO** |
| Zabezpečení dat | GDPR čl. 32 | **NESPLNĚNO** |
| Záznamy o činnostech zpracování | GDPR čl. 30 | **NESPLNĚNO** |
| Balanční test (oprávněný zájem) | GDPR čl. 6(1)(f) | **NEPROVEDEN** |
| Provedení DPIA | GDPR čl. 35 | **NEPROVEDENA** |

### 8.3 Doporučená doba uchovávání

ÚOOÚ doporučuje pro kamerové systémy maximální dobu uchovávání **72 hodin až 7 dní** (v závislosti na účelu). Projekt uchovává data **neomezeně**.

---

## 9. Licence trénovacích dat

### 9.1 Vlastní dataset (dataset/)

Projekt obsahuje vlastní dataset obrázků a anotací pro trénování detektoru SPZ.

**Právní otázky:**
- Obsahují obrázky reálné registrační značky? Pokud ano, jedná se o osobní údaje i v trénovacím datasetu
- Jak byly obrázky pořízeny? Je nutný souhlas/oprávněný zájem
- Jsou obrázky autorsky chráněny? Kdo je autorem?
- Je dataset řádně anonymizován?

**Stav: NEPOSOUZENO** - je třeba provést analýzu obsahu datasetu.

### 9.2 COCO dataset (pre-trained YOLOv8n)

- Licence: **CC BY 4.0** (Creative Commons Attribution)
- Povinnosti: Atribuce autorů datasetu
- Stav v projektu: **Atribuce chybí**

---

## 10. Copyright analýza

### 10.1 Autorství projektu

```
Copyright (c) 2026 Václav Dominik Štraser
Copyright (c) 2026 Antonín Majer
Copyright (c) 2026 Ondřej Czadek
```

Copyright notices všech tří spoluautorů jsou řádně uvedeny v souboru `LICENSE`.

### 10.2 Problém: Zkopírovaný kód SORT

Složka `sort/` obsahuje kompletní kopii SORT repozitáře (Alex Bewley, 2016-2020). Jedná se o legitimní distribuci pod GPL-3.0, ale:
- Není uvedeno, že se jedná o třetí stranu
- Není uveden autor v hlavním README
- Kód nebyl modifikován (což je v souladu s GPL § 4)

### 10.3 Duplicitní kód

Funkce v `server.py` a `main.py` obsahují duplicitní implementace (make_plate_rectangle, remove_blue_strip, OCR funkce, vote_readings). Jedná se o kód od stejného autora - bez právního problému, ale s technickým dluhem.

---

## 11. Přehled rizik a doporučení

### 11.1 Kritická rizika (vyžadují okamžitou akci)

#### R-01: Relicencování projektu na AGPL-3.0

**Problém:** Projekt nelze legálně distribuovat pod MIT licencí.

**Řešení - varianta A (doporučená pro open-source):**
1. Změnit licenci projektu z MIT na **AGPL-3.0**
2. Aktualizovat soubor `LICENSE` s textem AGPL-3.0
3. Přidat AGPL-3.0 header do všech zdrojových souborů
4. Zajistit, že webové rozhraní nabízí odkaz na zdrojový kód (požadavek AGPL § 13)
5. Získat souhlas všech přispěvatelů s relicencováním

**Řešení - varianta B (pro komerční použití):**
1. Zakoupit komerční licenci od **Ultralytics** (eliminuje AGPL povinnost)
2. Kontaktovat Alex Bewley (alex@bewley.ai) pro komerční licenci na SORT (viz SORT README: *"If you require a permissive license contact Alex"*)
3. Poté je možné ponechat MIT licenci na vlastním kódu
4. Odhadované náklady: Ultralytics Enterprise licence (individuální cenová nabídka)

**Řešení - varianta C (technická náhrada):**
1. Nahradit Ultralytics YOLOv8 alternativou s permisivní licencí (např. detekce přes PyTorch + vlastní model)
2. Nahradit SORT vlastní implementací trackeru nebo použít tracker s permisivní licencí (např. ByteTrack - MIT)
3. Poté je MIT licence legitimní

#### R-02: GDPR compliance framework

**Problém:** Zpracování osobních údajů bez právního základu.

**Řešení:**
1. Definovat a zdokumentovat právní základ zpracování (pravděpodobně oprávněný zájem - čl. 6(1)(f))
2. Provést balanční test oprávněného zájmu
3. Provést DPIA (Data Protection Impact Assessment)
4. Vytvořit záznamy o činnostech zpracování (čl. 30)
5. Implementovat informační povinnost (informační cedule u kamery)
6. Implementovat data retention policy (automatické mazání po 72h - 7 dnech)
7. Implementovat technická opatření (šifrování, autentizace - viz Security Audit)
8. Vytvořit privacy policy

### 11.2 Vysoká rizika

#### R-03: Vytvoření THIRD_PARTY_LICENSES souboru

Vytvořit soubor `THIRD_PARTY_LICENSES.md` s přehledem všech závislostí, jejich licencí a copyright notices. Minimálně:

```
SORT - GPL-3.0 - Copyright (C) 2016-2020 Alex Bewley
Ultralytics YOLOv8 - AGPL-3.0 - Copyright (C) Ultralytics Inc.
PyTorch - BSD-3-Clause - Copyright (c) Facebook, Inc.
OpenCV - Apache-2.0 - Copyright (C) OpenCV team
EasyOCR - Apache-2.0 - Copyright (C) JaidedAI
PaddleOCR - Apache-2.0 - Copyright (C) Baidu Inc.
docTR - Apache-2.0 - Copyright (C) Mindee
TrOCR - MIT - Copyright (C) Microsoft Corporation
Flask - BSD-3-Clause - Copyright (C) Pallets Projects
NumPy - BSD-3-Clause - Copyright (C) NumPy Developers
SciPy - BSD-3-Clause - Copyright (C) SciPy Developers
Pandas - BSD-3-Clause - Copyright (C) Pandas Development Team
Pillow - HPND - Copyright (C) Jeffrey A. Clark
filterpy - MIT - Copyright (C) Roger R Labbe Jr.
lap - BSD-2-Clause - Copyright (C) Tomas Kazmar
COCO dataset - CC BY 4.0 - Copyright (C) COCO Consortium
```

#### R-04: AGPL § 13 - Network Interaction Clause

Pokud bude projekt licencován pod AGPL-3.0, webové rozhraní **musí** nabízet uživatelům přístup ke zdrojovému kódu. Implementovat:

```html
<!-- Přidat do index.html -->
<footer>
    <a href="/source">Zdrojový kód (AGPL-3.0)</a>
</footer>
```

Nebo endpoint:
```python
@app.route('/source')
def source_code():
    return redirect('https://github.com/...')  # URL repozitáře
```

#### R-05: Posouzení datasetu z hlediska GDPR

Ověřit, zda dataset (`dataset/images/`) obsahuje reálné registrační značky, a pokud ano:
- Anonymizovat (rozmazat/nahradit) reálné SPZ v trénovacích datech
- Nebo zajistit právní základ pro jejich zpracování v trénovacím datasetu

### 11.3 Střední rizika

#### R-06: Dokumentace licenčních podmínek

Aktualizovat `README.md` o:
- Jasné uvedení licence projektu
- Odkaz na THIRD_PARTY_LICENSES
- Citace SORT a YOLOv8 (pokud je to research projekt)
- Upozornění na AGPL povinnosti pro downstream uživatele

#### R-07: Úplný requirements.txt

Doplnit `requirements.txt` o všechny skutečné závislosti s přesnými verzemi, aby bylo jasné, jaké licencované komponenty projekt využívá.

---

## 12. Matice licenční kompatibility

```
Licence projektu          Závislost              Kompatibilní?
──────────────────────────────────────────────────────────────
MIT (současná)      ←     GPL-3.0 (SORT)         NE ✗
MIT (současná)      ←     AGPL-3.0 (Ultralytics) NE ✗
AGPL-3.0 (navržená) ←    GPL-3.0 (SORT)         ANO ✓
AGPL-3.0 (navržená) ←    AGPL-3.0 (Ultralytics) ANO ✓
AGPL-3.0 (navržená) ←    Apache-2.0 (ostatní)   ANO ✓
AGPL-3.0 (navržená) ←    BSD-3-Clause (ostatní)  ANO ✓
AGPL-3.0 (navržená) ←    MIT (ostatní)           ANO ✓
```

**Jedinou licencí, která je kompatibilní se všemi závislostmi projektu, je AGPL-3.0.**

---

## 13. Souhrn povinností při relicencování na AGPL-3.0

Pokud bude projekt relicencován na AGPL-3.0, musí splňovat:

| Povinnost | Zdroj | Popis |
|-----------|-------|-------|
| Zdrojový kód dostupný přes síť | AGPL-3.0 § 13 | Webové rozhraní musí nabízet odkaz na zdrojový kód |
| Zachování copyright notices | AGPL-3.0 § 4 | Zachovat všechny stávající copyright notices |
| Označení modifikací | AGPL-3.0 § 5(a) | Prominentní oznámení, co bylo modifikováno |
| Celé dílo pod AGPL | AGPL-3.0 § 5(c) | Celý projekt musí být AGPL-3.0 |
| Distribuce zdrojového kódu | AGPL-3.0 § 6 | Zdrojový kód musí být dostupný příjemcům |
| Žádné další omezení | AGPL-3.0 § 10 | Nelze přidávat další licenční omezení |
| GPL-3.0 kompatibilita | AGPL-3.0 § 13 | SORT kód zůstává pod GPL-3.0 |

---

## 14. Přehled sankcí a rizik

### 14.1 Licenční porušení

| Riziko | Pravděpodobnost | Dopad |
|--------|-----------------|-------|
| Žaloba od Ultralytics (AGPL porušení) | Nízká (školní projekt) | Střední - soudní spor |
| Žaloba od Alex Bewley (GPL porušení) | Velmi nízká | Nízký |
| Ztráta práva na distribuci | Střední | Vysoký - musí se stáhnout |
| Reputační poškození | Nízká | Nízký |

### 14.2 GDPR porušení

| Riziko | Právní základ | Maximální sankce |
|--------|---------------|------------------|
| Zpracování bez právního základu | Čl. 83(5)(a) GDPR | Až 20 000 000 EUR / 4 % obratu |
| Nesplnění informační povinnosti | Čl. 83(5)(b) GDPR | Až 20 000 000 EUR / 4 % obratu |
| Nesplnění práv subjektu údajů | Čl. 83(5)(b) GDPR | Až 20 000 000 EUR / 4 % obratu |
| Neprovedení DPIA | Čl. 83(4)(a) GDPR | Až 10 000 000 EUR / 2 % obratu |
| Nedostatečné zabezpečení | Čl. 83(4)(a) GDPR | Až 10 000 000 EUR / 2 % obratu |

**Poznámka:** Pro školní/výzkumný projekt je pravděpodobnost sankcí nízká, ale ÚOOÚ může zahájit řízení i z vlastní iniciativy na základě podnětu.

---

## 15. Závěr

### Hlavní zjištění

1. **Licenční konflikt je kritický:** Projekt **nelze legálně distribuovat** pod MIT licencí kvůli zahrnutí GPL-3.0 (SORT) a závislosti na AGPL-3.0 (Ultralytics). Je nutné buď relicencovat na AGPL-3.0, nebo zakoupit komerční licence a nahradit copyleft komponenty.

2. **GDPR nesoulad je kritický:** Zpracování registračních značek bez právního základu, informační povinnosti, DPIA a technických opatření představuje porušení GDPR.

3. **Atribuční povinnosti nejsou splněny:** Chybí THIRD_PARTY_LICENSES soubor a citace.

### Doporučený postup (pro školní/open-source projekt)

1. **Okamžitě:** Změnit licenci na AGPL-3.0
2. **Okamžitě:** Přidat THIRD_PARTY_LICENSES soubor
3. **Okamžitě:** Přidat odkaz na zdrojový kód do webového rozhraní (AGPL § 13)
4. **Krátkodobě:** Provést DPIA a definovat právní základ zpracování
5. **Krátkodobě:** Implementovat data retention policy
6. **Střednědobě:** Implementovat technická bezpečnostní opatření (viz Security Audit)

---

**Upozornění:** Tento dokument představuje technickou analýzu licencí a regulatorního rámce. Nejedná se o právní poradenství. Pro závazné právní posouzení se obraťte na advokáta specializovaného na IT právo a ochranu osobních údajů.
