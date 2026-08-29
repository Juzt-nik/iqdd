"""
Unified inference service.

Loads whichever trained artifacts are present (tree model, CNN, and/or
autoencoder — the service degrades gracefully if only a subset was
trained) and produces one structured result per image:

  - Ensembled quality_score / quality_label
  - Per-issue type/severity/confidence (union across models, taking the
    stronger signal when both agree an issue is present)
  - The raw interpretable engineered features (for the API response /
    frontend "image statistics" panel)
  - A base64-encoded heatmap image (Grad-CAM + autoencoder reconstruction
    error, combined) for defect localization / explainability
  - A short natural-language explanation per flagged issue, grounded in
    the specific feature value that triggered it — not just "the model
    said so"

This module is deliberately self-contained (no FastAPI/DB imports) so it
can be exercised directly from a script or notebook, independent of the
web layer.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from app.ml.dataset import IMAGENET_MEAN, IMAGENET_STD
from app.ml.degrade import ISSUE_TYPES, ISSUE_SEVERITY_WEIGHTS
from app.ml.features import extract_features, feature_vector, FEATURE_NAMES
from app.ml.gradcam import GradCAM
from app.ml.model_cnn import ConvAutoencoder, QualityNet
from app.ml.model_tree import TreeQualityModel

# Thresholds for turning an engineered feature into a plain-language
# explanation. Calibrated against the synthetic degradation ranges in
# app/ml/degrade.py, not arbitrary.
_EXPLANATION_RULES = {
    "blur": lambda f: f"sharpness is low (Laplacian variance {f.laplacian_variance:.1f})",
    "underexposure": lambda f: f"image is dark ({f.underexposed_frac*100:.0f}% of pixels near-black, mean luminance {f.mean_luminance:.0f}/255)",
    "overexposure": lambda f: f"image is overexposed ({f.overexposed_frac*100:.0f}% of pixels clipped near-white)",
    "noise": lambda f: f"estimated noise level is elevated (sigma \u2248 {f.noise_sigma:.1f})",
    "corruption": lambda f: f"unusual block-edge/compression artifacts detected (blockiness {f.blockiness:.2f})",
    "defect": lambda f: "localized region flagged as anomalous relative to typical clean images",
}


def _overlay_heatmap(image_bgr: np.ndarray, heatmap01: np.ndarray) -> np.ndarray:
    """Blend a [0,1] heatmap onto the original image (for display)."""
    hm = (heatmap01 * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    hm_color = cv2.resize(hm_color, (image_bgr.shape[1], image_bgr.shape[0]))
    return cv2.addWeighted(image_bgr, 0.6, hm_color, 0.4, 0)


def _encode_png_base64(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("failed to encode heatmap image")
    return base64.b64encode(buf.tobytes()).decode("ascii")


class InferenceService:
    def __init__(self, model_dir: Path, device: str | None = None):
        self.model_dir = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tree_model: TreeQualityModel | None = self._try_load_tree()
        self.cnn_model, self.cnn_config, self.feat_mean, self.feat_std = self._try_load_cnn()
        self.autoencoder, self.anomaly_stats = self._try_load_autoencoder()
        self.gradcam: GradCAM | None = None
        if self.cnn_model is not None:
            self.gradcam = GradCAM(self.cnn_model, self.cnn_model.features[-1])

        if self.tree_model is None and self.cnn_model is None:
            raise RuntimeError(f"No trained models found under {self.model_dir}")

    # ---------- model loading (each optional; missing = skipped, not fatal) ----------

    def _try_load_tree(self):
        d = self.model_dir / "tree"
        if not (d / "score_model.joblib").exists():
            return None
        return TreeQualityModel.load(d)

    def _try_load_cnn(self):
        d = self.model_dir / "cnn"
        ckpt = d / "quality_net_best.pt"
        if not ckpt.exists():
            return None, None, None, None
        config = json.load(open(d / "config.json"))
        model = QualityNet(pretrained=False, feature_dim=len(config["feature_names"]))
        model.load_state_dict(torch.load(ckpt, map_location=self.device))
        model.to(self.device).eval()
        stats = np.load(d / "feature_norm_stats.npz")
        return model, config, stats["mean"], stats["std"]

    def _try_load_autoencoder(self):
        d = self.model_dir / "cnn"
        ckpt = d / "autoencoder_best.pt"
        if not ckpt.exists():
            return None, None
        ae = ConvAutoencoder()
        ae.load_state_dict(torch.load(ckpt, map_location=self.device))
        ae.to(self.device).eval()
        stats_path = d / "anomaly_stats.json"
        stats = json.load(open(stats_path)) if stats_path.exists() else None
        return ae, stats

    # ---------- preprocessing ----------

    def _preprocess_for_cnn(self, img_bgr: np.ndarray, img_size: int) -> torch.Tensor:
        img = cv2.resize(img_bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        t = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
        return t.to(self.device)

    def _preprocess_for_ae(self, img_bgr: np.ndarray, img_size: int = 128) -> torch.Tensor:
        img = cv2.resize(img_bgr, (img_size, img_size), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
        return t.to(self.device)

    # ---------- main entry point ----------

    def analyze(self, img_bgr: np.ndarray) -> dict:
        feats = extract_features(img_bgr)
        fvec = feature_vector(feats)

        tree_out, cnn_out, ae_out = None, None, None

        if self.tree_model is not None:
            tree_out = self.tree_model.predict(fvec)

        cnn_score, cnn_issue_probs, cam = None, None, None
        if self.cnn_model is not None:
            img_size = self.cnn_config["img_size"]
            img_t = self._preprocess_for_cnn(img_bgr, img_size)
            norm_feats = (fvec - self.feat_mean) / (self.feat_std + 1e-6)
            feats_t = torch.from_numpy(norm_feats.astype(np.float32)).unsqueeze(0).to(self.device)

            with torch.no_grad():
                score_t, logits_t = self.cnn_model(img_t, feats_t)
            cnn_score = float(score_t.item())
            cnn_issue_probs = torch.sigmoid(logits_t)[0].detach().cpu().numpy()

            # Grad-CAM w.r.t. the strongest predicted issue (falls back to
            # the score itself if nothing crosses 0.5), for the heatmap.
            top_idx = int(np.argmax(cnn_issue_probs))
            target = "issue" if cnn_issue_probs[top_idx] >= 0.5 else "score"
            feats_t_grad = feats_t.clone().requires_grad_(False)
            img_t_grad = img_t.clone().requires_grad_(True)
            cam = self.gradcam(img_t_grad, feats_t_grad,
                                target=target, issue_idx=top_idx if target == "issue" else None)
            cam = cam[0].cpu().numpy()

        anomaly_score, anomaly_prob, anomaly_map = None, None, None
        if self.autoencoder is not None:
            ae_t = self._preprocess_for_ae(img_bgr)
            with torch.no_grad():
                anomaly_score = float(self.autoencoder.anomaly_score(ae_t)[0].item())
                anomaly_map = self.autoencoder.anomaly_map(ae_t)[0].cpu().numpy()
            if self.anomaly_stats:
                z = (anomaly_score - self.anomaly_stats["clean_val_mean"]) / self.anomaly_stats["clean_val_std"]
                anomaly_prob = float(1 / (1 + np.exp(-0.75 * z)))  # sigmoid squashing of the z-score

        result = self._ensemble(feats, tree_out, cnn_score, cnn_issue_probs, anomaly_prob)

        heatmap_b64 = None
        if cam is not None or anomaly_map is not None:
            combined = np.zeros(img_bgr.shape[:2], dtype=np.float32)
            h, w = combined.shape
            if cam is not None:
                combined = np.maximum(combined, cv2.resize(cam, (w, h)))
            if anomaly_map is not None:
                am = anomaly_map / (anomaly_map.max() + 1e-8)
                combined = np.maximum(combined, cv2.resize(am, (w, h)))
            overlay = _overlay_heatmap(img_bgr, combined)
            heatmap_b64 = _encode_png_base64(overlay)

        result["heatmap_png_base64"] = heatmap_b64
        result["features"] = feats.to_dict()
        result["model_versions"] = {
            "tree": self.tree_model is not None,
            "cnn": self.cnn_model is not None,
            "autoencoder": self.autoencoder is not None,
        }
        return result

    # ---------- ensembling ----------

    def _ensemble(self, feats, tree_out, cnn_score, cnn_issue_probs, anomaly_prob) -> dict:
        scores = []
        if tree_out is not None:
            scores.append(tree_out["quality_score"])
        if cnn_score is not None:
            scores.append(cnn_score)
        quality_score = float(np.mean(scores)) if scores else 50.0

        # Per-issue confidence: max across whichever models produced a
        # signal for that issue type (a conservative "if either model is
        # confident, flag it" policy, appropriate for a quality gate).
        issue_conf: dict[str, float] = {}
        if tree_out is not None:
            for item in tree_out["issues"]:
                issue_conf[item["type"]] = max(issue_conf.get(item["type"], 0.0), item["confidence"])
        if cnn_issue_probs is not None:
            for i, t in enumerate(ISSUE_TYPES):
                if t == "defect":
                    continue  # defect handled via the autoencoder below, not the CNN head
                p = float(cnn_issue_probs[i])
                if p >= 0.5:
                    issue_conf[t] = max(issue_conf.get(t, 0.0), p)
        if anomaly_prob is not None and anomaly_prob >= 0.5:
            issue_conf["defect"] = max(issue_conf.get("defect", 0.0), anomaly_prob)

        issues = []
        for issue_type, conf in sorted(issue_conf.items(), key=lambda kv: -kv[1]):
            severity = "high" if conf > 0.85 else "medium" if conf > 0.65 else "low"
            explanation = _EXPLANATION_RULES.get(issue_type, lambda f: "")(feats)
            issues.append({
                "type": issue_type,
                "severity": severity,
                "confidence": round(conf, 3),
                "explanation": explanation,
            })

        # Consistency check between the score regressor(s) and the issue
        # classifiers: recompute an independent, penalty-based score using
        # the SAME per-issue weights the synthetic ground truth was
        # defined with (ISSUE_SEVERITY_WEIGHTS), treating each issue's
        # detection confidence as a proxy for its severity. This is an
        # approximation — confidence (how sure the classifier is an issue
        # exists) is not literally the same thing as severity (how bad
        # the degradation is) — but it's the only severity-like signal
        # available at inference time for classifier-based issues, and
        # using the same weights as training keeps the two scores from
        # talking past each other. We blend rather than let either one
        # override the other, since each captures something the other
        # can miss: the regressor sees the whole image directly, while
        # this catches cases where a classifier is confident about
        # something the regressor under-weighted.
        if issues:
            penalty = sum(ISSUE_SEVERITY_WEIGHTS[i["type"]] * i["confidence"] for i in issues)
            penalty_based_score = float(np.clip(100 - penalty, 0, 100))
            quality_score = 0.6 * quality_score + 0.4 * penalty_based_score
        quality_score = float(np.clip(quality_score, 0, 100))

        if quality_score >= 80:
            label = "ACCEPTABLE"
        elif quality_score >= 50:
            label = "DEGRADED"
        else:
            label = "DEFECTIVE"

        return {
            "quality_score": round(quality_score, 1),
            "quality_label": label,
            "issues": issues,
        }
