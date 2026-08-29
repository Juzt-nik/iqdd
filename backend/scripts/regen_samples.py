"""
Regenerate the samples/ folder from a REAL photo instead of the synthetic
placeholder graphic. Applies each degradation function in isolation (no
mixing) at a fixed severity, so the 7 outputs are directly comparable —
same source image, one variable changed at a time.

Usage (from backend/):
    python scripts/regen_samples.py --source data/clean/<some_real_photo>.jpg --output ../samples
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml.degrade import apply_blur, apply_underexposure, apply_overexposure, apply_noise, apply_corruption, apply_defect  # noqa: E402

# name, function, severity -- matches the severities documented in the
# original samples/README.md so the writeup stays accurate.
VARIANTS = [
    ("sample_1_clean", None, None),
    ("sample_2_blur", apply_blur, 0.75),
    ("sample_3_underexposed", apply_underexposure, 0.7),
    ("sample_4_overexposed", apply_overexposure, 0.6),
    ("sample_5_noise", apply_noise, 0.6),
    ("sample_6_corruption", apply_corruption, 0.7),
    ("sample_7_defect", apply_defect, 0.6),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path, help="A real clean photo")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    img = cv2.imread(str(args.source))
    if img is None:
        raise SystemExit(f"Could not read {args.source}")

    args.output.mkdir(parents=True, exist_ok=True)

    for name, fn, severity in VARIANTS:
        out_img = img if fn is None else fn(img, severity)
        out_path = args.output / f"{name}.jpg"
        cv2.imwrite(str(out_path), out_img)
        print(f"  wrote {out_path}")

    print(f"\nDone. Regenerated {len(VARIANTS)} samples in {args.output} from {args.source.name}")


if __name__ == "__main__":
    main()
