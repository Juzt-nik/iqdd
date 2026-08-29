"""PyTorch Dataset over the synthetic manifest for CNN training."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from app.ml.degrade import ISSUE_TYPES
from app.ml.features import FEATURE_NAMES

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class QualityDataset(Dataset):
    def __init__(self, manifest_df: pd.DataFrame, images_dir: Path,
                 feature_mean: np.ndarray, feature_std: np.ndarray, img_size: int = 128):
        self.df = manifest_df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(self.images_dir / row["filename"]))
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float()

        raw_feats = row[FEATURE_NAMES].values.astype(np.float32)
        norm_feats = (raw_feats - self.feature_mean) / (self.feature_std + 1e-6)
        feats_t = torch.from_numpy(norm_feats).float()

        score_t = torch.tensor(row["quality_score"], dtype=torch.float32)
        issue_labels = np.array([1.0 if row[f"issue_{t}"] > 0.05 else 0.0 for t in ISSUE_TYPES],
                                 dtype=np.float32)
        issues_t = torch.from_numpy(issue_labels)

        return img_t, feats_t, score_t, issues_t


def compute_feature_stats(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std of engineered features on the TRAIN split only, to avoid
    leaking val/test statistics into normalization."""
    X = df[FEATURE_NAMES].values.astype(np.float32)
    return X.mean(axis=0), X.std(axis=0)


class CleanImageDataset(Dataset):
    """For autoencoder training: clean images only, no labels needed."""

    def __init__(self, image_paths: list[Path], img_size: int = 128):
        self.paths = image_paths
        self.img_size = img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.paths[idx]))
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # [0,1] for Sigmoid decoder output
        return torch.from_numpy(img.transpose(2, 0, 1)).float()
