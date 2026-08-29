"""
Train QualityNet (CNN + engineered-feature hybrid) and the clean-image
autoencoder, then evaluate on the held-out test split.

Usage:
    python scripts/train_cnn.py \
        --manifest data/synthetic/manifest.csv \
        --images_dir data/synthetic/images \
        --clean_dir data/clean \
        --output data/models/cnn \
        --epochs 15 --batch_size 32
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                              precision_score, recall_score, roc_auc_score,
                              confusion_matrix)
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.dataset import QualityDataset, CleanImageDataset, compute_feature_stats
from app.ml.degrade import ISSUE_TYPES
from app.ml.model_cnn import QualityNet, ConvAutoencoder
from app.ml.features import FEATURE_NAMES


def train_quality_net(train_df, val_df, test_df, images_dir, output_dir,
                       epochs, batch_size, img_size, device, pretrained):
    feature_mean, feature_std = compute_feature_stats(train_df)

    train_ds = QualityDataset(train_df, images_dir, feature_mean, feature_std, img_size)
    val_ds = QualityDataset(val_df, images_dir, feature_mean, feature_std, img_size)
    test_ds = QualityDataset(test_df, images_dir, feature_mean, feature_std, img_size)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = QualityNet(pretrained=pretrained, feature_dim=len(FEATURE_NAMES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for img, feats, score, issues in train_dl:
            img, feats, score, issues = img.to(device), feats.to(device), score.to(device), issues.to(device)
            opt.zero_grad()
            pred_score, pred_logits = model(img, feats)
            loss = mse(pred_score, score) + 5.0 * bce(pred_logits, issues)
            loss.backward()
            opt.step()
            train_loss += loss.item() * img.size(0)
        train_loss /= len(train_ds)
        sched.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img, feats, score, issues in val_dl:
                img, feats, score, issues = img.to(device), feats.to(device), score.to(device), issues.to(device)
                pred_score, pred_logits = model(img, feats)
                loss = mse(pred_score, score) + 5.0 * bce(pred_logits, issues)
                val_loss += loss.item() * img.size(0)
        val_loss /= max(len(val_ds), 1)

        print(f"[QualityNet] epoch {epoch+1}/{epochs}  train_loss={train_loss:.3f}  val_loss={val_loss:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "quality_net_best.pt")

    model.load_state_dict(torch.load(output_dir / "quality_net_best.pt", map_location=device))
    report = evaluate_quality_net(model, test_dl, device)

    np.savez(output_dir / "feature_norm_stats.npz", mean=feature_mean, std=feature_std)
    with open(output_dir / "config.json", "w") as f:
        json.dump({"img_size": img_size, "feature_names": FEATURE_NAMES,
                    "issue_types": ISSUE_TYPES, "pretrained": pretrained}, f, indent=2)

    return model, report


@torch.no_grad()
def evaluate_quality_net(model, test_dl, device) -> dict:
    model.eval()
    all_scores_true, all_scores_pred = [], []
    all_issues_true, all_issues_prob = [], []

    for img, feats, score, issues in test_dl:
        img, feats = img.to(device), feats.to(device)
        pred_score, pred_logits = model(img, feats)
        all_scores_true.append(score.numpy())
        all_scores_pred.append(pred_score.cpu().numpy())
        all_issues_true.append(issues.numpy())
        all_issues_prob.append(torch.sigmoid(pred_logits).cpu().numpy())

    y_score = np.concatenate(all_scores_true)
    p_score = np.concatenate(all_scores_pred)
    y_issues = np.concatenate(all_issues_true)
    p_issues = np.concatenate(all_issues_prob)

    report = {"score_mae": float(mean_absolute_error(y_score, p_score)), "n_samples": len(y_score), "issues": {}}
    for i, t in enumerate(ISSUE_TYPES):
        yt, pp = y_issues[:, i], p_issues[:, i]
        yp = (pp >= 0.5).astype(int)
        if len(set(yt.astype(int))) < 2:
            report["issues"][t] = {"note": "skipped (degenerate class balance in this split)"}
            continue
        report["issues"][t] = {
            "accuracy": float(accuracy_score(yt, yp)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "roc_auc": float(roc_auc_score(yt, pp)),
            "confusion_matrix": confusion_matrix(yt, yp).tolist(),
            "positive_count": int(yt.sum()),
        }
    return report


def train_autoencoder(clean_dir, output_dir, epochs, batch_size, img_size, device):
    clean_paths = sorted(Path(clean_dir).glob("*.jpg")) + sorted(Path(clean_dir).glob("*.png"))
    n_val = max(1, int(0.15 * len(clean_paths)))
    val_paths, train_paths = clean_paths[:n_val], clean_paths[n_val:]

    train_ds = CleanImageDataset(train_paths, img_size)
    val_ds = CleanImageDataset(val_paths, img_size)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    ae = ConvAutoencoder().to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()

    best_val = float("inf")
    for epoch in range(epochs):
        ae.train()
        tr_loss = 0.0
        for x in train_dl:
            x = x.to(device)
            opt.zero_grad()
            recon = ae(x)
            loss = loss_fn(recon, x)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(train_ds)

        ae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x in val_dl:
                x = x.to(device)
                recon = ae(x)
                val_loss += loss_fn(recon, x).item() * x.size(0)
        val_loss /= max(len(val_ds), 1)

        print(f"[Autoencoder] epoch {epoch+1}/{epochs}  train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ae.state_dict(), output_dir / "autoencoder_best.pt")

    # Calibrate anomaly scores: compute per-image reconstruction error on
    # held-out CLEAN validation images so inference can convert a raw
    # error into a probability via (error - mean) / std, instead of an
    # arbitrary hardcoded threshold.
    ae.load_state_dict(torch.load(output_dir / "autoencoder_best.pt", map_location=device))
    ae.eval()
    val_scores = []
    with torch.no_grad():
        for x in val_dl:
            x = x.to(device)
            val_scores.append(ae.anomaly_score(x).cpu().numpy())
    val_scores = np.concatenate(val_scores) if val_scores else np.array([0.0])
    anomaly_stats = {
        "clean_val_mean": float(val_scores.mean()),
        "clean_val_std": float(val_scores.std() + 1e-6),
        "clean_val_p95": float(np.percentile(val_scores, 95)),
    }
    with open(output_dir / "anomaly_stats.json", "w") as f:
        json.dump(anomaly_stats, f, indent=2)
    print(f"[Autoencoder] calibration stats on clean val set: {anomaly_stats}")

    return ae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--images_dir", required=True, type=Path)
    ap.add_argument("--clean_dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--ae_epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--no_pretrained", action="store_true",
                     help="Skip ImageNet-pretrained backbone weights (for quick smoke tests without a download)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    df = pd.read_csv(args.manifest)
    train_df = df[df.split == "train"].reset_index(drop=True)
    val_df = df[df.split == "val"].reset_index(drop=True)
    test_df = df[df.split == "test"].reset_index(drop=True)
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    args.output.mkdir(parents=True, exist_ok=True)

    _, qnet_report = train_quality_net(
        train_df, val_df, test_df, args.images_dir, args.output,
        args.epochs, args.batch_size, args.img_size, device, pretrained=not args.no_pretrained)

    print("\n--- QualityNet test report ---")
    print(json.dumps(qnet_report, indent=2))

    train_autoencoder(args.clean_dir, args.output, args.ae_epochs, args.batch_size, args.img_size, device)

    with open(args.output / "eval_report.json", "w") as f:
        json.dump(qnet_report, f, indent=2)

    print(f"\nSaved CNN + autoencoder models and eval report to {args.output}")


if __name__ == "__main__":
    main()
