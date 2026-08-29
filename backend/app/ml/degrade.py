"""
Controlled synthetic degradation pipeline.

We don't have a real "quality-graded" dataset with ground-truth labels, so
per the assessment's section 8 we generate one: take clean images and apply
randomized, parameterized degradations, and use the known parameters as
ground truth for quality_score / issue-type / severity supervision.

Each degradation function takes a clean BGR uint8 image and a severity in
[0, 1] and returns (degraded_image, quality_score_delta, severity_label).
Severity label bucketing: 0.0-0.25 none, 0.25-0.5 low, 0.5-0.75 medium,
0.75-1.0 high (kept consistent with the API's severity vocabulary).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import cv2
import numpy as np

ISSUE_TYPES = ["blur", "underexposure", "overexposure", "noise", "corruption", "defect"]

# Per-issue penalty weights used to define the synthetic ground-truth
# quality_score (see DegradationResult.quality_score below) AND, for
# consistency, reused by the inference ensemble to convert a set of
# detected issues + confidences back into a score-consistency check.
# Single source of truth so the two don't drift apart.
ISSUE_SEVERITY_WEIGHTS = {"blur": 45, "underexposure": 35, "overexposure": 35,
                           "noise": 40, "corruption": 70, "defect": 60}


def _severity_label(sev: float) -> str:
    if sev < 0.05:
        return "none"
    if sev < 0.35:
        return "low"
    if sev < 0.65:
        return "medium"
    return "high"


@dataclass
class DegradationResult:
    image: np.ndarray
    applied: dict = field(default_factory=dict)  # issue_type -> severity (0-1)

    @property
    def quality_score(self) -> float:
        """
        Ground-truth quality score in [0, 100]: start clean at 100 and
        subtract a weighted penalty per applied issue, saturating so
        multiple simultaneous issues can still floor near 0.
        """
        penalty = sum(ISSUE_SEVERITY_WEIGHTS[k] * v for k, v in self.applied.items())
        return float(np.clip(100 - penalty, 0, 100))

    @property
    def quality_label(self) -> str:
        s = self.quality_score
        if s >= 80:
            return "ACCEPTABLE"
        if s >= 50:
            return "DEGRADED"
        return "DEFECTIVE"


def apply_blur(img: np.ndarray, severity: float) -> np.ndarray:
    ksize = int(1 + severity * 14) | 1  # odd kernel, up to ~15px
    if ksize <= 1:
        return img
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=severity * 6)


def apply_underexposure(img: np.ndarray, severity: float) -> np.ndarray:
    gamma = 1.0 + severity * 3.5  # gamma > 1 darkens
    factor = np.clip(1.0 - severity * 0.6, 0.15, 1.0)
    inv = 1.0 / gamma
    table = ((np.arange(256) / 255.0) ** inv * 255 * factor).astype(np.uint8)
    return cv2.LUT(img, table)


def apply_overexposure(img: np.ndarray, severity: float) -> np.ndarray:
    add = severity * 160
    return np.clip(img.astype(np.float32) + add, 0, 255).astype(np.uint8)


def apply_noise(img: np.ndarray, severity: float) -> np.ndarray:
    sigma = severity * 45
    gauss = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + gauss
    if severity > 0.5:  # add speckle for higher severities, common in sensor/low-light noise
        speckle = img.astype(np.float32) * np.random.normal(0, severity * 0.3, img.shape)
        noisy += speckle
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_corruption(img: np.ndarray, severity: float) -> np.ndarray:
    """Simulates severe degradation / partial corruption: block dropout,
    row shifts, or heavy JPEG re-encoding artifacts."""
    out = img.copy()
    h, w = out.shape[:2]
    mode = random.choice(["blocks", "jpeg", "channel_shift"])
    if mode == "blocks":
        n_blocks = int(severity * 12)
        for _ in range(n_blocks):
            bw, bh = random.randint(8, max(9, w // 8)), random.randint(8, max(9, h // 8))
            x, y = random.randint(0, max(0, w - bw)), random.randint(0, max(0, h - bh))
            out[y:y + bh, x:x + bw] = random.choice([0, 255])
    elif mode == "jpeg":
        quality = int(max(1, 45 - severity * 40))
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR if out.ndim == 3 else cv2.IMREAD_GRAYSCALE)
    else:
        shift = int(severity * 30)
        if out.ndim == 3:
            out[:, :, 0] = np.roll(out[:, :, 0], shift, axis=1)
            out[:, :, 2] = np.roll(out[:, :, 2], -shift, axis=1)
    return out


def apply_defect(img: np.ndarray, severity: float) -> np.ndarray:
    """Simulates a localized visual defect (scratch/spot/occlusion) rather
    than a global degradation — trains the model to distinguish 'globally
    degraded' from 'locally defective'."""
    out = img.copy()
    h, w = out.shape[:2]
    n = 1 + int(severity * 4)
    for _ in range(n):
        kind = random.choice(["scratch", "spot", "occlusion"])
        color = random.choice([(0, 0, 0), (255, 255, 255)]) if out.ndim == 3 else random.choice([0, 255])
        if kind == "scratch":
            p1 = (random.randint(0, w), random.randint(0, h))
            p2 = (random.randint(0, w), random.randint(0, h))
            cv2.line(out, p1, p2, color, thickness=max(1, int(severity * 4)))
        elif kind == "spot":
            center = (random.randint(0, w), random.randint(0, h))
            r = int(5 + severity * 25)
            cv2.circle(out, center, r, color, -1)
        else:
            bw, bh = int(w * severity * 0.3), int(h * severity * 0.3)
            x, y = random.randint(0, max(0, w - bw)), random.randint(0, max(0, h - bh))
            cv2.rectangle(out, (x, y), (x + bw, y + bh), color, -1)
    return out


DEGRADATION_FNS = {
    "blur": apply_blur,
    "underexposure": apply_underexposure,
    "overexposure": apply_overexposure,
    "noise": apply_noise,
    "corruption": apply_corruption,
    "defect": apply_defect,
}


def random_degrade(
    img: np.ndarray,
    max_issues: int = 2,
    p_clean: float = 0.12,
    rng: random.Random | None = None,
) -> DegradationResult:
    """
    Apply zero, one, or several randomly chosen degradations to a clean
    image, simulating realistic mixed-issue photos (e.g. blurry AND
    underexposed). Mutually-exclusive pairs (under/over-exposure) are
    avoided.
    """
    rng = rng or random
    if rng.random() < p_clean:
        return DegradationResult(image=img.copy(), applied={})

    n_issues = rng.randint(1, max_issues)
    candidates = list(DEGRADATION_FNS.keys())
    chosen = []
    while len(chosen) < n_issues and candidates:
        pick = rng.choice(candidates)
        candidates.remove(pick)
        if pick == "overexposure" and "underexposure" in chosen:
            continue
        if pick == "underexposure" and "overexposure" in chosen:
            continue
        chosen.append(pick)

    out = img.copy()
    applied = {}
    for issue in chosen:
        severity = rng.uniform(0.2, 1.0)
        out = DEGRADATION_FNS[issue](out, severity)
        applied[issue] = severity

    return DegradationResult(image=out, applied=applied)
