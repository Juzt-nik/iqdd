"""
Generate a labeled image-quality dataset from a folder of clean images by
applying randomized synthetic degradations (see app/ml/degrade.py).

Usage:
    python scripts/generate_dataset.py \
        --input_dir data/clean \
        --output_dir data/synthetic \
        --per_image 4 \
        --img_size 256

Produces:
    data/synthetic/images/*.jpg          (degraded images)
    data/synthetic/manifest.csv          (labels + engineered features + split)

Ground truth per row: quality_score, quality_label, one column per issue
type holding its applied severity (0.0 if not applied), and a `split`
column (train/val/test) assigned by clean-image identity so no image's
variants leak across splits.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.degrade import random_degrade, ISSUE_TYPES
from app.ml.features import extract_features, FEATURE_NAMES

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in VALID_EXTS)


def assign_split(image_id: str, val_frac=0.15, test_frac=0.15) -> str:
    """Deterministic split by hashing the source image name, so all
    degraded variants of one clean image stay in the same split."""
    h = abs(hash(image_id)) % 1000 / 1000.0
    if h < test_frac:
        return "test"
    if h < test_frac + val_frac:
        return "val"
    return "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--per_image", type=int, default=4,
                     help="Number of degraded variants to generate per clean image")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    images_out = args.output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    clean_paths = list_images(args.input_dir)
    if not clean_paths:
        raise SystemExit(f"No images found under {args.input_dir}")
    print(f"Found {len(clean_paths)} clean images. Generating {args.per_image} variants each "
          f"-> {len(clean_paths) * args.per_image} total samples.")

    manifest_path = args.output_dir / "manifest.csv"
    fieldnames = (["filename", "source_image", "split", "quality_score", "quality_label"]
                  + [f"issue_{t}" for t in ISSUE_TYPES] + FEATURE_NAMES)

    n_written = 0
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for clean_path in tqdm(clean_paths, desc="Degrading images"):
            img = cv2.imread(str(clean_path))
            if img is None:
                continue  # skip unreadable source files
            img = cv2.resize(img, (args.img_size, args.img_size), interpolation=cv2.INTER_AREA)
            split = assign_split(clean_path.stem)

            for k in range(args.per_image):
                rng = random.Random(f"{clean_path.stem}_{k}_{args.seed}")
                result = random_degrade(img, rng=rng)
                out_name = f"{clean_path.stem}_{k}.jpg"
                cv2.imwrite(str(images_out / out_name), result.image,
                             [cv2.IMWRITE_JPEG_QUALITY, 95])

                feats = extract_features(result.image)
                row = {
                    "filename": out_name,
                    "source_image": clean_path.name,
                    "split": split,
                    "quality_score": round(result.quality_score, 2),
                    "quality_label": result.quality_label,
                    **{f"issue_{t}": round(result.applied.get(t, 0.0), 4) for t in ISSUE_TYPES},
                    **feats.to_dict(),
                }
                row.pop("decodable", None)
                writer.writerow(row)
                n_written += 1

    print(f"Wrote {n_written} samples to {manifest_path}")


if __name__ == "__main__":
    main()
