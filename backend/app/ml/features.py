"""
Classical, interpretable image-quality feature extraction.

These features are:
  1. Used directly as rule-based / thresholded signals for explainability
     (each ties to a specific issue type).
  2. Fed as auxiliary input into the learned model (hybrid approach).

All functions accept a BGR or grayscale numpy array (OpenCV convention)
and return plain Python floats/dicts so results are JSON-serializable.
"""
from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass, asdict


@dataclass
class QualityFeatures:
    # --- Sharpness / blur ---
    laplacian_variance: float          # higher = sharper
    tenengrad: float                   # gradient-magnitude based sharpness

    # --- Exposure ---
    mean_luminance: float              # 0-255
    underexposed_frac: float           # fraction of pixels in low bins
    overexposed_frac: float            # fraction of pixels in high (clipped) bins

    # --- Noise ---
    noise_sigma: float                 # estimated noise std (robust MAD-based)

    # --- Contrast / dynamic range ---
    luminance_std: float
    dynamic_range: float               # p99 - p1 percentile of luminance

    # --- Color ---
    colorfulness: float
    saturation_mean: float

    # --- Compression / corruption ---
    blockiness: float                  # JPEG 8x8 block-edge energy
    decodable: bool                    # False if image failed to decode/read
    entropy: float                     # Shannon entropy of luminance histogram

    def to_dict(self) -> dict:
        return asdict(self)


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def laplacian_variance(gray: np.ndarray) -> float:
    """Classic blur metric: variance of the Laplacian. Low = blurry."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def tenengrad(gray: np.ndarray) -> float:
    """Gradient-magnitude based sharpness (Sobel energy)."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx ** 2 + gy ** 2))


def exposure_stats(gray: np.ndarray) -> tuple[float, float, float]:
    """Returns (mean_luminance, underexposed_frac, overexposed_frac)."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = hist.sum()
    under = hist[:10].sum() / total   # near-black bins
    over = hist[246:].sum() / total   # near-white / clipped bins
    return float(gray.mean()), float(under), float(over)


def estimate_noise_sigma(gray: np.ndarray) -> float:
    """
    Robust noise estimator: convolve with a Laplacian-like kernel that
    suppresses image structure but responds strongly to noise, then take
    a MAD-based (outlier-robust) estimate of the residual standard
    deviation. Based on the Immerkaer (1996) fast noise estimation idea,
    with a median-absolute-deviation normalizer instead of a plain mean
    so real edges (which produce large but sparse responses) don't
    inflate the estimate the way they would with a mean-based version.
    """
    M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = cv2.filter2D(gray.astype(np.float64), -1, M)
    kernel_gain = np.sqrt(np.sum(M ** 2))  # = 6 for this kernel; convolving
    # iid noise of std sigma through M scales output std by this factor,
    # so we must divide it back out to recover sigma itself.
    sigma = float(np.median(np.abs(conv)) / 0.6745 / kernel_gain)
    return max(0.0, sigma)


def dynamic_range(gray: np.ndarray) -> tuple[float, float]:
    std = float(gray.std())
    p1, p99 = np.percentile(gray, [1, 99])
    return std, float(p99 - p1)


def colorfulness_metric(img_bgr: np.ndarray) -> float:
    """Hasler & Susstrunk colorfulness metric."""
    if img_bgr.ndim != 3:
        return 0.0
    B, G, R = img_bgr[:, :, 0].astype(float), img_bgr[:, :, 1].astype(float), img_bgr[:, :, 2].astype(float)
    rg = R - G
    yb = 0.5 * (R + G) - B
    std_rg, mean_rg = rg.std(), rg.mean()
    std_yb, mean_yb = yb.std(), yb.mean()
    std_root = np.sqrt(std_rg ** 2 + std_yb ** 2)
    mean_root = np.sqrt(mean_rg ** 2 + mean_yb ** 2)
    return float(std_root + 0.3 * mean_root)


def saturation_mean(img_bgr: np.ndarray) -> float:
    if img_bgr.ndim != 3:
        return 0.0
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean())


def blockiness_metric(gray: np.ndarray) -> float:
    """
    Estimate JPEG block-edge energy: compare gradient magnitude at 8-pixel
    grid boundaries vs. interior gradients. High ratio => visible blocking.
    """
    gray = gray.astype(np.float64)
    h, w = gray.shape
    if h < 16 or w < 16:
        return 0.0
    gx = np.abs(np.diff(gray, axis=1))
    grid_cols = gx[:, 7:w - 1:8]
    if grid_cols.size == 0:
        return 0.0
    grid_energy = grid_cols.mean()
    overall_energy = gx.mean() + 1e-6
    return float(grid_energy / overall_energy)


def shannon_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-9)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def extract_features(img_bgr: np.ndarray) -> QualityFeatures:
    """Main entry point: compute the full interpretable feature set."""
    gray = _to_gray(img_bgr)
    lap_var = laplacian_variance(gray)
    ten = tenengrad(gray)
    mean_lum, under_frac, over_frac = exposure_stats(gray)
    noise = estimate_noise_sigma(gray)
    lum_std, dyn_range = dynamic_range(gray)
    colorfulness = colorfulness_metric(img_bgr)
    sat = saturation_mean(img_bgr)
    block = blockiness_metric(gray)
    entropy = shannon_entropy(gray)

    return QualityFeatures(
        laplacian_variance=lap_var,
        tenengrad=ten,
        mean_luminance=mean_lum,
        underexposed_frac=under_frac,
        overexposed_frac=over_frac,
        noise_sigma=noise,
        luminance_std=lum_std,
        dynamic_range=dyn_range,
        colorfulness=colorfulness,
        saturation_mean=sat,
        blockiness=block,
        decodable=True,
        entropy=entropy,
    )


def feature_vector(feats: QualityFeatures) -> np.ndarray:
    """Numeric vector (fixed order) for feeding into the learned model."""
    return np.array([
        feats.laplacian_variance,
        feats.tenengrad,
        feats.mean_luminance,
        feats.underexposed_frac,
        feats.overexposed_frac,
        feats.noise_sigma,
        feats.luminance_std,
        feats.dynamic_range,
        feats.colorfulness,
        feats.saturation_mean,
        feats.blockiness,
        feats.entropy,
    ], dtype=np.float32)


FEATURE_NAMES = [
    "laplacian_variance", "tenengrad", "mean_luminance", "underexposed_frac",
    "overexposed_frac", "noise_sigma", "luminance_std", "dynamic_range",
    "colorfulness", "saturation_mean", "blockiness", "entropy",
]
