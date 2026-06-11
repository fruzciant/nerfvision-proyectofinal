"""
split_dataset.py — Organiza el dataset en train/val (aleatorio 80/20) para YOLO.

Situación: Label Studio exportó las etiquetas (labels/*.txt) pero NO las
imágenes, y además añadió un prefijo hash a cada .txt:
    c8616938-tipoA1_f0003.txt   ->  imagen real: tipoA1_f0003.jpg

Este script:
  1. Lee cada .txt de LABELS_DIR y le quita el prefijo hash.
  2. Busca la imagen correspondiente en FRAMES_DIR (data/frames_raw).
  3. Copia el par (imagen + etiqueta) a data/dataset/ con split 80/20.

Uso:
    python scripts/split_dataset.py
"""

import random
import shutil
from pathlib import Path

LABELS_DIR = Path("/home/fruzciant/Downloads/data_export/labels/")   # .txt exportados de Label Studio
FRAMES_DIR = Path("/home/fruzciant/Documents/Universidad/visionComputador/nerfvision-proyectofinal/data/frames_raw/")      # imágenes originales extraídas
DST = Path("data/dataset")                # salida para YOLO

VAL_FRACTION = 0.20
SEED = 42
IMG_EXTS = [".jpg", ".jpeg", ".png"]

random.seed(SEED)

for split in ["train", "val"]:
    (DST / "images" / split).mkdir(parents=True, exist_ok=True)
    (DST / "labels" / split).mkdir(parents=True, exist_ok=True)


def strip_prefix(txt_stem):
    """Quita el prefijo hash de Label Studio: 'c8616938-tipoA1_f0003' -> 'tipoA1_f0003'."""
    # El prefijo es un hash seguido de '-'. Quitamos todo hasta el primer '-'.
    if "-" in txt_stem:
        return txt_stem.split("-", 1)[1]
    return txt_stem


def find_image(real_name):
    """Busca la imagen real en FRAMES_DIR probando las extensiones."""
    for ext in IMG_EXTS:
        candidate = FRAMES_DIR / f"{real_name}{ext}"
        if candidate.exists():
            return candidate
    return None


pairs = []         # (imagen_origen, etiqueta_origen, nombre_real)
missing = []
for label in LABELS_DIR.glob("*.txt"):
    real_name = strip_prefix(label.stem)
    img = find_image(real_name)
    if img is None:
        missing.append(real_name)
        continue
    pairs.append((img, label, real_name))

if missing:
    print(f"⚠ {len(missing)} etiquetas sin imagen encontrada (ejemplos): {missing[:5]}")

if not pairs:
    print("ERROR: no se emparejó ninguna imagen. Revisa FRAMES_DIR y los nombres.")
    raise SystemExit(1)

random.shuffle(pairs)
n_val = max(int(len(pairs) * VAL_FRACTION), 1)
val_set = pairs[:n_val]
train_set = pairs[n_val:]


def copy_pairs(group, split):
    for img, label, real_name in group:
        # Renombramos imagen y etiqueta al nombre real (sin hash), iguales entre sí
        shutil.copy(img, DST / "images" / split / f"{real_name}{img.suffix}")
        shutil.copy(label, DST / "labels" / split / f"{real_name}.txt")


copy_pairs(train_set, "train")
copy_pairs(val_set, "val")

print(f"Emparejadas: {len(pairs)} imágenes con etiqueta")
print(f"  train: {len(train_set)}")
print(f"  val:   {len(val_set)}")
print(f"Dataset listo en: {DST}/")
