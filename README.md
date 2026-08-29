# AI-Powered Image Quality & Defect Detection

A full-stack application that accepts an image and evaluates its visual
quality — detecting blur, under/overexposure, noise, corruption, and
localized visual defects — using a hybrid classical-CV + deep-learning
pipeline, with no external AI/vision APIs.

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
- [Before you submit](#before-you-submit)

## Architecture

```
┌──────────────┐      ┌──────────────────────┐      ┌────────────────────┐
│   Frontend    │ HTTP │       Backend         │      │   Inference core    │
│  React+Vite   │─────▶│  FastAPI + SQLAlchemy │─────▶│  app/ml/*            │
│  (nginx prod) │◀─────│  SQLite / Postgres    │◀─────│  tree + CNN + AE     │
└──────────────┘      └──────────────────────┘      └────────────────────┘
                                                              │
                                                     ┌────────┴────────┐
                                                     │  Trained models  │
                                                     │  data/models/    │
                                                     └──────────────────┘
```

- **Frontend**: React (Vite), talks to the backend via `/api/v1/*`. In
  Docker, nginx reverse-proxies these calls so the browser only ever sees
  one origin (no CORS in production).
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
parameters as ground truth. Quality score is defined as
`100 − Σ(issue_weight × severity)` for the applied issues (see
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
├── samples/                   sample images per quality condition
├── docker-compose.yml
└── .env.example
```

## Quickstart — Docker (recommended)

Requires trained models to already exist under `backend/data/models/` (see
[Training your own models](#training-your-own-models) — the repo ships
with a small validation run already in place so the stack is runnable
out of the box, but **you should retrain on real data before submitting**;
see [Before you submit](#before-you-submit)).

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
   Suggested: Kaggle's [`prasunroy/natural-images`](https://www.kaggle.com/datasets/prasunroy/natural-images)
   (~6,900 images, small, diverse — fast to iterate on). Drop them into
   `backend/data/clean/`.

2. **Generate the synthetic labeled dataset:**
   ```bash
   cd backend
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
   `--img_size 224`; on your RTX 5070, or Kaggle's T4):
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

Methodology: held-out test split (by source-image identity, so no
degraded variant of a test image appears in train/val), metrics computed
per issue type — accuracy, precision, recall, F1, ROC-AUC, confusion
matrix — plus MAE for the quality-score regression. Both training scripts
write these to `data/models/{tree,cnn}/eval_report.json` automatically.

> **⚠️ The numbers currently checked into this repo are from a tiny
> (40-image, procedurally-generated) proxy dataset used only to validate
> the pipeline end-to-end while building it — not a real evaluation.**
> Re-run `train_tree.py` and `train_cnn.py` against your real Kaggle-sourced
> dataset, then replace this section with the resulting numbers before
> submitting. Suggested content once you have real numbers:
> - A summary table: per-issue precision/recall/F1/ROC-AUC for both the
>   tree and CNN models side by side (useful ablation — shows what the
>   CNN adds over the engineered features alone).
> - The quality-score MAE for both models.
> - 3–5 concrete failure cases from the test set (image + predicted vs.
>   true label) with a sentence on why each is a plausible confusion —
>   e.g. a heavily-textured clean image that reads as "noisy" to the
>   classical noise estimator, or an aggressively color-graded photo that
>   trips the exposure heuristics despite being intentional.

## Explainability

Every `/analyze` response includes:
- `features` — the full interpretable feature vector for that image
- `issues[].explanation` — plain-language, feature-grounded reasoning per
  flagged issue
- `heatmap_png_base64` — Grad-CAM (CNN) blended with the autoencoder's
  reconstruction-error map, toggleable in the frontend as "Defect view"

## Limitations & failure cases

- **Confidence is used as a proxy for severity** in the ensembling step
  (how sure a classifier is an issue exists ≠ how bad that issue is).
  This is documented explicitly in `inference.py`'s `_ensemble` method;
  it's the best signal available at inference time for classifier-based
  issues, but it's an approximation worth stating plainly rather than
  glossing over.
- **Corruption is the hardest issue type** to detect reliably, because it
  spans three mechanically very different degradations (block dropout,
  JPEG re-encoding, channel shift) lumped under one label — worth
  reporting as a specific limitation, and a candidate for splitting into
  separate issue types if you have time.
- **The autoencoder's anomaly threshold is calibrated only against clean
  validation images** (mean + std of reconstruction error on held-out
  clean images, converted to a z-score-based probability). It hasn't been
  validated against a diverse "genuinely defective" test set, since the
  synthetic `defect` degradations are themselves the only defect-labeled
  data available.
- **Small/procedural training data checked into this repo currently** —
  see the callout in [Evaluation & results](#evaluation--results).

## Before you submit

- [ ] Retrain `tree` and `cnn` models on a real image dataset (see
      [Training your own models](#training-your-own-models))
- [ ] Replace the [Evaluation & results](#evaluation--results) section
      with real metrics + failure-case discussion
- [ ] Replace `samples/` with real photos per condition (see
      `samples/README.md`)
- [ ] Run `docker compose up --build` yourself end-to-end as a final
      sanity check — this was built and tested without a live Docker
      daemon available, so the compose/Dockerfile setup, while written
      carefully against standard patterns, hasn't been build-tested here
- [ ] If deploying online, add the URL to this README (optional per the
      brief — local Docker Compose is acceptable)
