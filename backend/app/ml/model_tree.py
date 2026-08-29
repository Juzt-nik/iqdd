"""
Tree-based model on engineered classical features (app/ml/features.py).

Two heads, sharing the same 12-dim feature vector as input:
  - `score_model`:  HistGradientBoostingRegressor  -> quality_score (0-100)
  - `issue_models`: one HistGradientBoostingClassifier per issue type
                     -> P(issue present), from which severity bucket is
                        derived by thresholding the predicted probability

This is intentionally simple and fast to train (seconds, not GPU-hours),
which makes it a good sanity-check baseline against the CNN, and a
reasonable fallback if the CNN underperforms or time runs short.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                              precision_score, recall_score, roc_auc_score,
                              confusion_matrix)

from app.ml.degrade import ISSUE_TYPES
from app.ml.features import FEATURE_NAMES


class TreeQualityModel:
    def __init__(self):
        self.score_model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=6, random_state=42)
        self.issue_models: dict[str, HistGradientBoostingClassifier] = {
            t: HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_depth=6, random_state=42)
            for t in ISSUE_TYPES
        }
        self.feature_names = FEATURE_NAMES

    def fit(self, df: pd.DataFrame):
        X = df[self.feature_names].values
        y_score = df["quality_score"].values
        self.score_model.fit(X, y_score)

        for t in ISSUE_TYPES:
            y_issue = (df[f"issue_{t}"] > 0.05).astype(int).values
            if y_issue.sum() == 0 or y_issue.sum() == len(y_issue):
                # Degenerate split (all-one-class) — skip fitting, fall
                # back to a constant predictor at inference time.
                self.issue_models[t] = None
                continue
            self.issue_models[t].fit(X, y_issue)
        return self

    def predict(self, feature_vector: np.ndarray) -> dict:
        X = feature_vector.reshape(1, -1)
        score = float(np.clip(self.score_model.predict(X)[0], 0, 100))
        issues = []
        for t in ISSUE_TYPES:
            model = self.issue_models[t]
            if model is None:
                continue
            proba = float(model.predict_proba(X)[0, 1])
            if proba >= 0.5:
                severity = "high" if proba > 0.85 else "medium" if proba > 0.65 else "low"
                issues.append({"type": t, "confidence": round(proba, 3), "severity": severity})
        return {"quality_score": round(score, 1), "issues": issues}

    def evaluate(self, df: pd.DataFrame) -> dict:
        """Compute regression + per-issue classification metrics on a
        held-out split, per assessment section 9."""
        X = df[self.feature_names].values
        y_score = df["quality_score"].values
        pred_score = self.score_model.predict(X)

        report = {
            "score_mae": float(mean_absolute_error(y_score, pred_score)),
            "n_samples": len(df),
            "issues": {},
        }

        for t in ISSUE_TYPES:
            model = self.issue_models[t]
            y_true = (df[f"issue_{t}"] > 0.05).astype(int).values
            if model is None or y_true.sum() == 0 or y_true.sum() == len(y_true):
                report["issues"][t] = {"note": "skipped (degenerate class balance in this split)"}
                continue
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)[:, 1]
            cm = confusion_matrix(y_true, y_pred).tolist()
            report["issues"][t] = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(set(y_true)) > 1 else None,
                "confusion_matrix": cm,
                "positive_count": int(y_true.sum()),
            }
        return report

    def save(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.score_model, path / "score_model.joblib")
        for t, m in self.issue_models.items():
            if m is not None:
                joblib.dump(m, path / f"issue_{t}.joblib")
        with open(path / "meta.json", "w") as f:
            json.dump({"feature_names": self.feature_names, "issue_types": ISSUE_TYPES}, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TreeQualityModel":
        path = Path(path)
        obj = cls()
        obj.score_model = joblib.load(path / "score_model.joblib")
        for t in ISSUE_TYPES:
            fp = path / f"issue_{t}.joblib"
            obj.issue_models[t] = joblib.load(fp) if fp.exists() else None
        return obj
