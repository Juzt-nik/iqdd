"""
Pick a stratified, reproducible subset of images from an extracted
Kaggle image-classification folder (one subfolder per class) to use as
the "clean" pool for generate_dataset.py.

Usage:
    python scripts/select_subset.py \
        --input_dir ~/Downloads/natural_images \
        --output_dir data/clean \
        --per_class 50 \
        --seed 42
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, type=Path,
                     help="Root folder with one subfolder per class")
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--per_class", type=int, default=50,
                     help="How many images to sample from each class")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    class_dirs = [d for d in args.input_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        raise SystemExit(f"No subfolders found in {args.input_dir}")

    total = 0
    for class_dir in sorted(class_dirs):
        images = [p for p in class_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        if not images:
            print(f"  {class_dir.name}: no images, skipping")
            continue
        k = min(args.per_class, len(images))
        chosen = random.sample(images, k)
        for src in chosen:
            dst = args.output_dir / f"{class_dir.name}_{src.name}"
            shutil.copy2(src, dst)
        total += k
        print(f"  {class_dir.name}: {k}/{len(images)} copied")

    print(f"\nTotal: {total} images -> {args.output_dir}")


if __name__ == "__main__":
    main()
