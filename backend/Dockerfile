# syntax=docker/dockerfile:1
FROM python:3.11-slim

# OpenCV needs these shared libs even in headless mode; libgomp1 is required
# by torch's threading backend.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install torch/torchvision CPU-only wheels first, from PyTorch's own index —
# the default PyPI wheels bundle CUDA and add ~2GB with no benefit if this
# container has no GPU. If you ARE deploying with GPU access, delete this
# block and just let requirements.txt (below) pull the CUDA build instead.
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu

# requirements.txt also pins torch/torchvision (so local, non-Docker setup
# on a GPU machine installs the right versions too) — strip them back out
# here rather than relying on pip to recognize the CPU build as already
# satisfying the plain version pin, since that behavior isn't worth
# depending on for a step this expensive to get wrong.
COPY requirements.txt .
RUN grep -v -E "^(torch|torchvision)==" requirements.txt > requirements.docker.txt \
    && pip install --no-cache-dir -r requirements.docker.txt

COPY app ./app
COPY data/models ./data/models

# Runtime-writable locations — mounted as volumes in docker-compose so
# analysis results and uploaded images survive container restarts.
RUN mkdir -p /app/storage/images

ENV PYTHONUNBUFFERED=1 \
    DATABASE_URL=sqlite:////app/storage/iqdd.db \
    STORAGE_DIR=/app/storage/images \
    MODEL_DIR=/app/data/models

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
