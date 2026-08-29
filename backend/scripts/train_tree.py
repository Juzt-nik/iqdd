"""
Train and evaluate the engineered-feature tree model on a manifest
produced by generate_dataset.py.

Usage:
    python scripts/train_tree.py \
        --manifest data/synthetic/manifest.csv \
        --output data/models/tree
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml.model_tree import TreeQualityModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    train_df, val_df, test_df = (df[df.split == s].reset_index(drop=True)
                                  for s in ("train", "val", "test"))
    print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    model = TreeQualityModel()
    model.fit(train_df)

    val_report = model.evaluate(val_df)
    test_report = model.evaluate(test_df)

    print("\n--- Validation ---")
    print(json.dumps(val_report, indent=2))
    print("\n--- Test (held-out) ---")
    print(json.dumps(test_report, indent=2))

    args.output.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    with open(args.output / "eval_report.json", "w") as f:
        json.dump({"validation": val_report, "test": test_report}, f, indent=2)
    print(f"\nSaved model + eval report to {args.output}")


if __name__ == "__main__":
    main()
