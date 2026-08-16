# =============================================================================
#  ATRIA EchoTrace
#
#  Two build targets from one file:
#    review  - slim, CPU-only, no torch. Dataset browsing, tracing, metrics,
#              figures and exports. ~250 MB.
#    ai      - adds torch/transformers/peft/bitsandbytes for local MedGemma
#              inference. Large (several GB) and needs the gated weights, which
#              are mounted from the host Hugging Face cache rather than baked in.
#
#  Build:  docker build -f docker/Dockerfile --target review -t atria:review .
#          docker build -f docker/Dockerfile --target ai     -t atria:ai .
# =============================================================================

# --- shared base -------------------------------------------------------------
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    ATRIA_HOST=0.0.0.0 \
    ATRIA_PORT=8000 \
    ATRIA_DATASET_DIR=/data/sample-dataset \
    ATRIA_OUTPUT_DIR=/outputs \
    HF_HOME=/hf-cache

WORKDIR /app

# libgomp is required by scikit-image/scipy wheels; curl is used by HEALTHCHECK.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

# Dependency metadata first so the layer caches independently of source edits.
COPY pyproject.toml README.md ./
COPY src ./src

# Run as a non-root user; outputs and the HF cache are writable mounts.
RUN useradd --create-home --uid 10001 atria \
 && mkdir -p /outputs /hf-cache /data \
 && chown -R atria:atria /app /outputs /hf-cache /data

EXPOSE 8000
VOLUME ["/outputs", "/hf-cache"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/meta/health || exit 1

# --- review tier -------------------------------------------------------------
FROM base AS review

RUN pip install --no-cache-dir . && chown -R atria:atria /app
USER atria
CMD ["python", "-m", "atria_echotrace", "serve", "--no-browser"]

# --- AI tier -----------------------------------------------------------------
FROM base AS ai

# CPU wheels by default: they are far smaller and work everywhere. For GPU, build
# with --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126 and run with
# `--gpus all` (see docker-compose.yml, profile "ai-gpu").
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --index-url "${TORCH_INDEX}" torch \
 && pip install --no-cache-dir ".[ai]" \
 && chown -R atria:atria /app
USER atria
CMD ["python", "-m", "atria_echotrace", "serve", "--no-browser"]
