"""
Trénovací skript pro vlastní YOLO model na detekci SPZ.

Použití:
    1. Stáhni dataset (např. z Roboflow) ve formátu YOLOv8
    2. Uprav cestu v DATA_YAML na svůj dataset.yaml
    3. Spusť: python train_model.py

Dataset musí mít strukturu:
    dataset/
    ├── data.yaml
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
"""
from ultralytics import YOLO
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Trénování YOLO modelu pro detekci SPZ')
    parser.add_argument('--data', type=str, default='./dataset/data.yaml',
                        help='Cesta k dataset YAML souboru')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Počet epoch (minimum 50)')
    parser.add_argument('--batch', type=int, default=16,
                        help='Velikost batche')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Velikost vstupního obrázku')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Výchozí YOLO model (pretrained)')
    parser.add_argument('--name', type=str, default='spz_detector',
                        help='Název běhu trénování')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.epochs < 50:
        print('[WARNING] Minimum je 50 epoch! Nastavuji na 50.')
        args.epochs = 50

    print('=' * 60)
    print('  TRÉNOVÁNÍ YOLO MODELU PRO DETEKCI SPZ')
    print('=' * 60)
    print(f'  Dataset:  {args.data}')
    print(f'  Epochy:   {args.epochs}')
    print(f'  Batch:    {args.batch}')
    print(f'  ImgSize:  {args.imgsz}')
    print(f'  Model:    {args.model}')
    print('=' * 60)

    # Načtení pretrained modelu
    model = YOLO(args.model)

    # Trénování
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        name=args.name,
        patience=0,         # Bez early stopping - proběhne všech N epoch
        save=True,          # Ukládání checkpointů
        save_period=10,     # Checkpoint každých 10 epoch
        plots=True,         # Generování grafů
        verbose=True,
    )

    print('\n' + '=' * 60)
    print('  TRÉNOVÁNÍ DOKONČENO')
    print('=' * 60)
    print(f'  Nejlepší model: runs/detect/{args.name}/weights/best.pt')
    print(f'  Poslední model: runs/detect/{args.name}/weights/last.pt')
    print('=' * 60)

    # Validace na test setu
    print('\n[INFO] Spouštím validaci na validačním setu...')
    metrics = model.val()
    print(f'  mAP50:    {metrics.box.map50:.4f}')
    print(f'  mAP50-95: {metrics.box.map:.4f}')

    # Export modelu
    print('\n[INFO] Exportuji model...')
    best_model = YOLO(f'runs/detect/{args.name}/weights/best.pt')
    best_model.export(format='torchscript')
    print('[INFO] Model exportován.')


if __name__ == '__main__':
    main()
