from __future__ import annotations

import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.database import get_db
from app.db.models import Analysis
from app.schemas.analysis import AnalysisListOut, AnalysisOut, AnalysisSummaryOut, HealthOut

router = APIRouter()


@router.get("/health", response_model=HealthOut, tags=["system"])
def health(request: Request, db: Session = Depends(get_db)):
    svc = getattr(request.app.state, "inference_service", None)
    db_ok = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    models_loaded = {"tree": False, "cnn": False, "autoencoder": False}
    if svc is not None:
        models_loaded = {
            "tree": svc.tree_model is not None,
            "cnn": svc.cnn_model is not None,
            "autoencoder": svc.autoencoder is not None,
        }
    status_str = "ok" if (svc is not None and db_ok) else "degraded"
    return HealthOut(status=status_str, models_loaded=models_loaded,
                      database="ok" if db_ok else "unreachable")


@router.post("/api/v1/analyze", response_model=AnalysisOut, status_code=status.HTTP_201_CREATED, tags=["analysis"])
async def analyze_image(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    svc = getattr(request.app.state, "inference_service", None)
    if svc is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Inference models are not loaded")

    if file.content_type not in settings.ALLOWED_CONTENT_TYPES:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                             f"Unsupported content type '{file.content_type}'. "
                             f"Allowed: {sorted(settings.ALLOWED_CONTENT_TYPES)}")

    raw = await file.read()
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                             f"File exceeds {settings.MAX_UPLOAD_MB}MB limit")
    if len(raw) == 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file is empty")

    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        # Explicitly satisfies section 5's "handle invalid or unreadable
        # images gracefully" — a corrupt/truncated/non-image file must
        # not crash the pipeline, and is itself meaningful signal (it's
        # about as "defective" as an image can get).
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                             "File could not be decoded as a valid image "
                             "(corrupt, truncated, or not an image file)")

    try:
        result = svc.analyze(img)
    except Exception as exc:  # noqa: BLE001 — surface as a clean 500, don't leak a stack trace to the client
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                             f"Analysis failed: {exc}") from exc

    image_id = str(uuid.uuid4())
    ext = Path(file.filename or "upload.jpg").suffix or ".jpg"
    stored_name = f"{image_id}{ext}"
    stored_path = settings.STORAGE_DIR / stored_name
    with open(stored_path, "wb") as f:
        f.write(raw)

    record = Analysis(
        id=image_id,
        original_filename=file.filename or "upload",
        stored_image_path=str(stored_path),
        quality_score=result["quality_score"],
        quality_label=result["quality_label"],
        issues=result["issues"],
        features=result["features"],
        model_versions=result["model_versions"],
        heatmap_png_base64=result.get("heatmap_png_base64"),
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return record.to_dict()


@router.get("/api/v1/analyses", response_model=AnalysisListOut, tags=["analysis"])
def list_analyses(
    db: Session = Depends(get_db),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    quality_label: str | None = Query(None, description="Filter by ACCEPTABLE/DEGRADED/DEFECTIVE"),
):
    q = db.query(Analysis)
    if quality_label:
        q = q.filter(Analysis.quality_label == quality_label.upper())
    total = q.count()
    rows = q.order_by(Analysis.created_at.desc()).offset(offset).limit(limit).all()

    results = [
        AnalysisSummaryOut(
            id=r.id, original_filename=r.original_filename, created_at=r.created_at,
            quality_score=r.quality_score, quality_label=r.quality_label,
            issue_types=[i["type"] for i in (r.issues or [])],
        )
        for r in rows
    ]
    return AnalysisListOut(total=total, limit=limit, offset=offset, results=results)


@router.get("/api/v1/analyses/{analysis_id}", response_model=AnalysisOut, tags=["analysis"])
def get_analysis(analysis_id: str, db: Session = Depends(get_db)):
    record = db.get(Analysis, analysis_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No analysis found with id '{analysis_id}'")
    return record.to_dict()


@router.get("/api/v1/analyses/{analysis_id}/image", tags=["analysis"])
def get_analysis_image(analysis_id: str, db: Session = Depends(get_db)):
    """Serves the originally uploaded image for a past analysis, so the
    frontend can show 'original vs. defect view' toggles on the history
    detail page (not just at the moment of upload)."""
    record = db.get(Analysis, analysis_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No analysis found with id '{analysis_id}'")
    path = Path(record.stored_image_path)
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Stored image file is missing on disk")
    return FileResponse(path)
