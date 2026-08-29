from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class IssueOut(BaseModel):
    type: str
    severity: str
    confidence: float
    explanation: str


class AnalysisOut(BaseModel):
    id: str
    original_filename: str
    created_at: datetime | None
    quality_score: float
    quality_label: str
    issues: list[IssueOut]
    features: dict[str, Any]
    model_versions: dict[str, bool]
    heatmap_png_base64: str | None = None


class AnalysisSummaryOut(BaseModel):
    """Lighter-weight shape for the history list endpoint — omits the
    heatmap (can be tens of KB each) and per-feature detail so listing
    many past analyses stays fast."""
    id: str
    original_filename: str
    created_at: datetime | None
    quality_score: float
    quality_label: str
    issue_types: list[str]


class AnalysisListOut(BaseModel):
    total: int
    limit: int
    offset: int
    results: list[AnalysisSummaryOut]


class ErrorOut(BaseModel):
    detail: str


class HealthOut(BaseModel):
    status: str
    models_loaded: dict[str, bool]
    database: str
