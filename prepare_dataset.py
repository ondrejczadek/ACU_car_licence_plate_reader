"""
Rozdeleni oanotovanych obrazku na train/valid set.

Pouziti:
    1. python prepare_dataset.py   (rozdeli na train/valid)
    2. python train_model.py --data ./dataset/data.yaml --epochs 50
"""
import os
import shutil
import glob
import random

images_dir = './dataset/images'
labels_dir = './dataset/labels'
train_img = './dataset/train/images'
train_lbl = './dataset/train/labels'
valid_img = './dataset/valid/images'
valid_lbl = './dataset/valid/labels'

VALID_SPLIT = 0.2  # 20% pro validaci

# find all images that have a matching label
image_files = sorted(glob.glob(os.path.join(images_dir, '*.png')) +
                     glob.glob(os.path.join(images_dir, '*.jpg')))

annotated = []
missing = []
for img_path in image_files:
    name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(labels_dir, name + '.txt')
    if os.path.exists(label_path):
        annotated.append((img_path, label_path))
    else:
        missing.append(img_path)

print(f'Celkem obrazku: {len(image_files)}')
print(f'S anotaci:      {len(annotated)}')
print(f'Bez anotace:    {len(missing)}')

if len(annotated) < 5:
    print('\nPrilis malo oanotovanych obrazku!')
    exit(1)

# shuffle and split
random.seed(42)
random.shuffle(annotated)

split_idx = max(1, int(len(annotated) * VALID_SPLIT))
valid_files = annotated[:split_idx]
train_files = annotated[split_idx:]

print(f'\nTrain: {len(train_files)} obrazku')
print(f'Valid: {len(valid_files)} obrazku')

# clean and copy
for d in [train_img, train_lbl, valid_img, valid_lbl]:
    os.makedirs(d, exist_ok=True)
    for f in glob.glob(os.path.join(d, '*')):
        os.remove(f)

for img_path, lbl_path in train_files:
    shutil.copy2(img_path, train_img)
    shutil.copy2(lbl_path, train_lbl)

for img_path, lbl_path in valid_files:
    shutil.copy2(img_path, valid_img)
    shutil.copy2(lbl_path, valid_lbl)

print(f'\nHotovo! Ted spust: python train_model.py --data ./dataset/data.yaml --epochs 50')
