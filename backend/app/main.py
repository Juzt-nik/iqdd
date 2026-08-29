from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.db.database import init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("iqdd")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create DB tables and load the inference models ONCE, so
    # per-request latency doesn't pay model-load cost on every call.
    init_db()
    try:
        from app.ml.inference import InferenceService
        app.state.inference_service = InferenceService(model_dir=settings.MODEL_DIR)
        logger.info("Inference service loaded: %s", {
            "tree": app.state.inference_service.tree_model is not None,
            "cnn": app.state.inference_service.cnn_model is not None,
            "autoencoder": app.state.inference_service.autoencoder is not None,
        })
    except Exception as exc:  # noqa: BLE001
        app.state.inference_service = None
        msg = f"Failed to load inference models from {settings.MODEL_DIR}: {exc}"
        if settings.REQUIRE_MODELS:
            logger.error(msg)
            raise RuntimeError(msg) from exc
        logger.warning("%s (continuing with REQUIRE_MODELS=false; /analyze will return 503)", msg)

    yield
    # Shutdown: nothing to clean up currently (DB sessions are per-request).


app = FastAPI(
    title="AI-Powered Image Quality & Defect Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
