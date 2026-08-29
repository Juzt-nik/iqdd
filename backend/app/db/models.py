from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Float, DateTime, JSON, Text

from app.db.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(String, primary_key=True, default=_uuid)
    original_filename = Column(String, nullable=False)
    stored_image_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    quality_score = Column(Float, nullable=False)
    quality_label = Column(String, nullable=False)

    # Stored as JSON blobs — SQLite/Postgres both support the JSON type
    # via SQLAlchemy; keeps the schema simple while retaining full
    # structured detail for the history/detail endpoints.
    issues = Column(JSON, nullable=False, default=list)
    features = Column(JSON, nullable=False, default=dict)
    model_versions = Column(JSON, nullable=False, default=dict)

    heatmap_png_base64 = Column(Text, nullable=True)

    def to_dict(self, include_heatmap: bool = True) -> dict:
        d = {
            "id": self.id,
            "original_filename": self.original_filename,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "quality_score": self.quality_score,
            "quality_label": self.quality_label,
            "issues": self.issues,
            "features": self.features,
            "model_versions": self.model_versions,
        }
        if include_heatmap:
            d["heatmap_png_base64"] = self.heatmap_png_base64
        return d
