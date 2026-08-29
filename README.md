# AI-Powered Image Quality & Defect Detection

A full-stack application that accepts an image and evaluates its visual
quality — detecting blur, under/overexposure, noise, corruption, and
localized visual defects — using a hybrid classical-CV + deep-learning
pipeline, with no external AI/vision APIs.

**Live deployment:**
- Frontend: https://iqdd.vercel.app
- Backend API: https://iqdd.onrender.com (health: https://iqdd.onrender.com/health)

> Both are on free-tier hosting. The backend sleeps after 15 minutes of
> inactivity — the first request after idle time can take 30-60 seconds to
> respond while it wakes up. This is expected free-tier behavior, not a bug.

## Contents

- [Architecture](#architecture)
- [AI/ML approach](#aiml-approach)
- [Project structure](#project-structure)
- [Quickstart — Docker](#quickstart--docker-recommended)
- [Quickstart — local development](#quickstart--local-development)
- [Training your own models](#training-your-own-models)
- [Database](#database)
- [API reference](#api-reference)
- [Evaluation & results](#evaluation--results)
- [Explainability](#explainability)
- [Limitations & failure cases](#limitations--failure-cases)
- [Deployment notes](#deployment-notes)

## Architecture

![System Architecture](public/architecture.png)

- **Frontend**: React (Vite), talks to the backend via `/api/v1/*`. In
  Docker (or same-origin deployments), nginx reverse-proxies these calls so
  the browser only ever sees one origin. In the cloud deployment above,
  frontend and backend are on separate origins (Vercel + Render), so the
  frontend is built with `VITE_API_BASE` pointing at the backend URL and
  the backend's `CORS_ORIGINS` allows the frontend's origin.
- **Backend**: FastAPI. Loads all trained models once at startup, exposes
  REST endpoints, persists results to a SQL database.
- **Inference core**: framework-agnostic Python (`app/ml/`) — no FastAPI or
  DB imports — so it can be driven from scripts/notebooks independent of
  the web layer.

## AI/ML approach

No external AI services are used. The pipeline is a **hybrid** of three
components, ensembled at inference time:

1. **Classical, interpretable features** (`app/ml/features.py`) — computed
   for every image regardless of which learned models are available:
   Laplacian-variance and Tenengrad sharpness, exposure histogram
   clipping fractions, a MAD-based noise-σ estimator, contrast/dynamic
   range, colorfulness/saturation, JPEG blockiness, and entropy. These
   double as the explainability layer (each flagged issue cites the
   specific feature value that triggered it) and as auxiliary input to
   the learned models below.

2. **Tree model** (`app/ml/model_tree.py`) — gradient-boosted trees
   (`HistGradientBoostingRegressor`/`Classifier`) on the engineered
   feature vector: one regressor for `quality_score`, one binary
   classifier per issue type. Fast to train (seconds), a good baseline
   and a fallback if the CNN underperforms or time runs short.

3. **CNN** (`app/ml/model_cnn.py`) — a MobileNetV3-small backbone
   (ImageNet-pretrained, downloaded once at training time via
   torchvision — this doesn't violate the "no external AI services"
   constraint, since nothing is called over the network at inference)
   fused with the engineered feature vector, with two heads: quality-score
   regression and multi-label issue classification.

4. **Autoencoder** (`app/ml/model_cnn.py::ConvAutoencoder`) — a small
   convolutional autoencoder trained **only on clean images**.
   Reconstruction error (both a scalar and a per-pixel map) is the signal
   for "potential visual defect" — the one issue type that's inherently
   about *local* anomalies rather than a *global* image-quality
   statistic, so it gets its own detector rather than being folded into
   the classifiers above.

**Training data**: there's no pre-existing "quality-graded" dataset with
ground truth, so one is generated synthetically (`scripts/generate_dataset.py`)
by applying randomized, parameterized degradations
(`app/ml/degrade.py`) to clean images, using the known degradation
parameters as ground truth. Source images are 3,600 real photos from
Kaggle's [`prasunroy/natural-images`](https://www.kaggle.com/datasets/prasunroy/natural-images)
dataset (8 classes: airplane, car, cat, dog, flower, fruit, motorbike,
person), each degraded 6 ways — 21,600 total samples. Quality score is
defined as `100 − Σ(issue_weight × severity)` for the applied issues (see
`ISSUE_SEVERITY_WEIGHTS` in `degrade.py`) — the same weights are reused at
inference time in the ensembling step, so the two stay consistent.
Train/val/test splits are assigned by clean-source-image identity (not
per-variant), so no image's degraded variants leak across splits.

**Ensembling** (`app/ml/inference.py::InferenceService`): the tree and CNN
score predictions are averaged; per-issue confidence takes the max across
whichever models produced a signal for that issue (a conservative
"either model confident enough → flag it" policy, appropriate for a
quality gate); the final score is a blend of the regressor-average and an
independent penalty-based score recomputed from the flagged issues using
the same severity weights the ground truth was defined with, so the score
and the issues list can't contradict each other.

**Explainability**: Grad-CAM (`app/ml/gradcam.py`) on the CNN's last
conv block, targeted at whichever issue it's most confident about;
combined with the autoencoder's reconstruction-error map into one overlay
heatmap. Every flagged issue also carries a plain-language explanation
grounded in its actual feature value (e.g. *"estimated noise level is
elevated (sigma ≈ 14.1)"*), not just a bare confidence number.

## Project structure

```
iqdd/
├── backend/
│   ├── app/
│   │   ├── main.py            FastAPI app, lifespan model loading
│   │   ├── api/routes.py      /analyze, /analyses, /analyses/{id}, /health
│   │   ├── core/config.py     env-var driven settings
│   │   ├── db/                SQLAlchemy models + session
│   │   ├── schemas/           Pydantic request/response models
│   │   └── ml/
│   │       ├── features.py    classical feature extraction
│   │       ├── degrade.py     synthetic degradation engine
│   │       ├── dataset.py     PyTorch Dataset classes
│   │       ├── model_tree.py  gradient-boosted tree model
│   │       ├── model_cnn.py   QualityNet CNN + ConvAutoencoder
│   │       ├── gradcam.py     Grad-CAM
│   │       └── inference.py   unified ensembled inference service
│   ├── scripts/
│   │   ├── select_subset.py   stratified sampler for source photo pool
│   │   ├── regen_samples.py   regenerates samples/ from a real photo
│   │   ├── generate_dataset.py
│   │   ├── train_tree.py
│   │   └── train_cnn.py
│   ├── data/
│   │   ├── clean/             ← put your clean source images here
│   │   ├── synthetic/         ← generated by generate_dataset.py
│   │   └── models/            ← trained model artifacts (tree/, cnn/)
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/                  React (Vite) SPA
│   ├── src/
│   ├── Dockerfile
│   └── nginx.conf
├── samples/                   sample images per quality condition (real photo)
├── docker-compose.yml
└── .env.example
```

## Quickstart — Docker (recommended)

```bash
docker compose up --build
```

- Frontend: http://localhost:8080
- Backend API: http://localhost:8000 (docs at http://localhost:8000/docs)
- Health check: http://localhost:8000/health

The backend won't accept traffic as "ready" (per its Docker health check)
until models finish loading — give it ~30–60s on first start. Uploaded
images and the SQLite database persist in a named Docker volume
(`backend_storage`) across restarts.

To retrain and pick up new models without rebuilding the image, drop new
artifacts into `backend/data/models/{tree,cnn}/` and run
`docker compose restart backend` (that directory is bind-mounted, not
baked into the image at runtime).

## Quickstart — local development

**Backend:**
```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend** (separate terminal):
```bash
cd frontend
npm install
npm run dev
```
Vite's dev server proxies `/api` and `/health` to `http://localhost:8000`
(see `vite.config.js`) — open http://localhost:5173.

## Training your own models

1. **Get clean source images.** Any reasonably diverse photo set works.
   This project used Kaggle's [`prasunroy/natural-images`](https://www.kaggle.com/datasets/prasunroy/natural-images)
   (~6,900 images across 8 classes). `scripts/select_subset.py` samples
   evenly across class subfolders so the pool stays visually diverse
   rather than skewed toward whichever class the source folder lists
   first:
   ```bash
   python scripts/select_subset.py \
       --input_dir /path/to/natural_images --output_dir data/clean --per_class 50
   ```

2. **Generate the synthetic labeled dataset:**
   ```bash
   python scripts/generate_dataset.py \
       --input_dir data/clean --output_dir data/synthetic \
       --per_image 6 --img_size 224
   ```
   Produces `data/synthetic/images/` and `data/synthetic/manifest.csv`
   (labels + engineered features + train/val/test split).

3. **Train the tree baseline** (fast, do this first as a sanity check):
   ```bash
   python scripts/train_tree.py \
       --manifest data/synthetic/manifest.csv --output data/models/tree
   ```

4. **Train the CNN + autoencoder** (needs a GPU to be practical at
   `--img_size 224` — this project trained on a Kaggle notebook's free T4):
   ```bash
   python scripts/train_cnn.py \
       --manifest data/synthetic/manifest.csv \
       --images_dir data/synthetic/images \
       --clean_dir data/clean \
       --output data/models/cnn \
       --epochs 25 --ae_epochs 25 --batch_size 64 --img_size 224
   ```
   Both scripts write an `eval_report.json` with held-out test-set metrics
   — see [Evaluation & results](#evaluation--results).

5. Restart the backend (or `docker compose restart backend`) to pick up
   the new artifacts.

6. **Regenerate sample images** for `samples/` from one real photo, showing
   each condition in isolation at a fixed severity:
   ```bash
   python scripts/regen_samples.py --source data/clean/<a_real_photo>.jpg --output ../samples
   ```

## Database

Defaults to a local SQLite file (`storage/iqdd.db` in Docker,
`./storage/iqdd.db` locally) — created automatically on first startup, no
setup needed.

To use Postgres instead, set `DATABASE_URL` (e.g.
`postgresql://user:pass@host:5432/iqdd`) and add `psycopg2-binary` to
`requirements.txt`; no other code changes are needed since the app goes
through SQLAlchemy's engine abstraction.

## API reference

Interactive docs (Swagger UI) are auto-generated by FastAPI at `/docs`
once the backend is running.

### `POST /api/v1/analyze`
Upload an image for analysis.

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@path/to/image.jpg;type=image/jpeg"
```

```json
{
  "id": "9ff448e8-6176-45f8-96d5-b15d85d62fac",
  "original_filename": "image.jpg",
  "created_at": "2026-08-28T10:42:31.967Z",
  "quality_score": 65.9,
  "quality_label": "DEGRADED",
  "issues": [
    {
      "type": "noise",
      "severity": "high",
      "confidence": 1.0,
      "explanation": "estimated noise level is elevated (sigma \u2248 14.1)"
    }
  ],
  "features": { "laplacian_variance": 4036.3, "noise_sigma": 14.08, "...": "..." },
  "model_versions": { "tree": true, "cnn": true, "autoencoder": true },
  "heatmap_png_base64": "iVBORw0KGgoAAAANSUhEUgA..."
}
```
Errors: `415` unsupported content type, `413` file too large, `400` empty
file, `422` file couldn't be decoded as a valid image, `503` models not
loaded.

### `GET /api/v1/analyses`
Paginated history. Query params: `limit` (default 20, max 100), `offset`,
`quality_label` (filter by `ACCEPTABLE`/`DEGRADED`/`DEFECTIVE`).

```bash
curl "http://localhost:8000/api/v1/analyses?limit=10&quality_label=DEGRADED"
```

### `GET /api/v1/analyses/{id}`
Full detail for one past analysis (same shape as the `analyze` response).

### `GET /api/v1/analyses/{id}/image`
Serves the originally uploaded image binary (used by the frontend's
history detail view for the "original vs. defect view" toggle).

### `GET /health`
```json
{"status": "ok", "models_loaded": {"tree": true, "cnn": true, "autoencoder": true}, "database": "ok"}
```

## Evaluation & results

Methodology: held-out test split (3,138 samples, by source-image identity
so no degraded variant of a test image appears in train/val), metrics
computed per issue type on 3,600 real source photos degraded 6 ways each
(21,600 total samples).

### Tree model (classical features, fast baseline)

| Issue | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Blur | 0.955 | 0.939 | 0.860 | 0.897 | 0.985 |
| Underexposure | 0.973 | 0.953 | 0.915 | 0.934 | 0.992 |
| Overexposure | 0.968 | 0.948 | 0.898 | 0.922 | 0.994 |
| Noise | 0.980 | 0.976 | 0.933 | 0.954 | 0.995 |
| Corruption | 0.867 | 0.817 | 0.576 | 0.676 | 0.897 |
| Defect | 0.895 | 0.827 | 0.681 | 0.747 | 0.925 |

Score MAE: 12.13 (0–100 scale).

### CNN (QualityNet)

| Issue | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Blur | 0.980 | 0.963 | 0.949 | 0.956 | 0.995 |
| Underexposure | 0.977 | 0.977 | 0.909 | 0.942 | 0.989 |
| Overexposure | 0.970 | 0.962 | 0.893 | 0.926 | 0.992 |
| Noise | 0.988 | 0.995 | 0.948 | 0.971 | 0.997 |
| Corruption | 0.971 | 0.984 | 0.895 | 0.937 | 0.991 |
| Defect | 0.971 | 0.968 | 0.902 | 0.934 | 0.979 |

Score MAE: 4.56.

### Autoencoder (anomaly detector)

Trained on clean images only; anomaly score is reconstruction error.
Reference stats from held-out clean validation images: mean 0.0334, std
0.0082, 95th percentile 0.0486. At inference, a new image's reconstruction
error is z-scored against these to flag "defect" as an anomaly relative to
what clean images normally look like, independent of the supervised
defect classifier above — the two are combined at inference for the final
defect call.

### Reading these results

- **CNN outperforms the tree model on every issue**, most sharply on
  **corruption** (F1 0.68 → 0.94) and **defect** (F1 0.75 → 0.93). This is
  expected: corruption and defects are spatial/structural patterns (block
  dropout, scratches, localized occlusions) that a handful of global image
  statistics can't capture well, while a CNN can learn spatial filters
  for them directly. The tree model's weaker recall specifically (0.58 and
  0.68) means it's missing real cases more than it's over-flagging — it's
  the higher-value model to fall back on only when the CNN is unavailable.
- Score MAE drops from 12.13 (tree) to 4.56 (CNN) on the same 0–100 scale,
  consistent with the same pattern.
- Validation and test metrics track closely across both models (no large
  gap), which is the main evidence against overfitting to the synthetic
  generation process itself, given both splits come from the same
  generator.

### Failure cases & limitations

- **Out-of-distribution inputs are misjudged with high confidence.**
  Testing with non-photographic inputs — a procedurally-generated
  gradient/shapes graphic, and separately a system-architecture diagram
  screenshot — both got flagged as "defective" by the autoencoder at
  ~99% confidence, and the diagram was additionally flagged "overexposure"
  at 100% confidence (81% of its pixels are genuinely near-white, since
  it's a mostly-white diagram — the feature measurement is accurate, but
  the semantic label "overexposure" assumes photographic content that
  isn't present). Both models were trained exclusively on natural
  photographs, so non-photographic input is far outside the training
  distribution — the models don't know what they don't know, and
  currently report high confidence rather than uncertainty on such inputs.
  This is a known limitation of the current confidence outputs, not a bug:
  a genuine out-of-distribution detector (e.g. thresholding the
  autoencoder's reconstruction error against the clean-validation stats
  above, independent of the labeled classifiers) would be the next step to
  address this.
- **Corruption and defect detection are the weakest categories for both
  models**, though the CNN closes most of the gap. These issue types are
  inherently harder to separate from natural image variation (e.g. a
  genuinely dark, textured, or high-contrast photo can resemble a
  synthetic corruption artifact) and are the categories most likely to
  produce false negatives in practice.
- **All training and evaluation data is synthetically degraded**, not
  real-world defective photos with human-verified labels. The
  degradation parameters are known and controlled, so metrics reflect how
  well the models recover those known parameters — not necessarily how
  well they'd generalize to naturally-occurring defects (e.g. real lens
  scratches, real sensor noise from actual low-light conditions) that
  weren't explicitly modeled by the degradation functions in
  `app/ml/degrade.py`.
- **Confidence is used as a proxy for severity** in the ensembling step
  (how sure a classifier is an issue exists ≠ how bad that issue is).
  This is documented explicitly in `inference.py`'s `_ensemble` method;
  it's the best signal available at inference time for classifier-based
  issues, but it's an approximation worth stating plainly rather than
  glossing over.
- **Corruption spans three mechanically different degradations** (block
  dropout, JPEG re-encoding, channel shift) lumped under one label — a
  candidate for splitting into separate issue types with more time.
- **The autoencoder's anomaly threshold is calibrated only against clean
  validation images** (mean + std of reconstruction error on held-out
  clean images, converted to a z-score-based probability). It hasn't been
  validated against a diverse "genuinely defective" test set, since the
  synthetic `defect` degradations are themselves the only defect-labeled
  data available.

## Explainability

Every `/analyze` response includes:
- `features` — the full interpretable feature vector for that image
- `issues[].explanation` — plain-language, feature-grounded reasoning per
  flagged issue
- `heatmap_png_base64` — Grad-CAM (CNN) blended with the autoencoder's
  reconstruction-error map, toggleable in the frontend as "Defect view"

## Deployment notes

Deployed as two separate services (rather than the same-origin Docker
Compose setup, which is also fully supported and was verified locally
first):

- **Backend**: [Render](https://render.com), free tier, deployed directly
  from `backend/Dockerfile` (the same image used and verified in the local
  Docker Compose setup). Free-tier constraints: 512MB RAM (comfortably fits
  this app's CPU-only torch + opencv + model footprint, verified in
  production), and the service sleeps after 15 minutes idle (30-60s
  cold-start on the next request). Environment variables are set in
  Render's dashboard, mirroring `docker-compose.yml`.
- **Frontend**: [Vercel](https://vercel.com), built from `frontend/`
  with `VITE_API_BASE` set at build time to the Render backend URL, since
  the two services are on different origins (no shared nginx reverse proxy
  like the Docker setup). The backend's `CORS_ORIGINS` is set to the
  Vercel domain to allow this.
- **Model loading**: both deployments load `data/models/{tree,cnn}/` once
  at FastAPI startup (`app/main.py`'s lifespan handler) — the same code
  path locally, in Docker, and in the cloud. `/health` reports whether
  each of the three models (tree, CNN, autoencoder) loaded successfully.
