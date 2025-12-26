# syntax=docker/dockerfile:1.4
################################################################################
# SpeciesNet Server Dockerfile
# Uses the speciesnet library to serve detection/classification via HTTP
################################################################################
FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim AS builder

WORKDIR /app

ENV UV_LINK_MODE=copy

# Native Python deps (e.g. ml-dtypes, torch) may need a compiler toolchain
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential

# Create virtual environment and install speciesnet with server extras
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python "speciesnet[server]"

################################################################################
# Final image
################################################################################
FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim AS runtime

WORKDIR /app

# Runtime libs needed by OpenCV (cv2) and matplotlib (for fonts)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    fontconfig \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Copy the synced virtualenv from the builder
COPY --from=builder /app/.venv /app/.venv

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Model cache directory - mount a volume here for persistence
ENV HF_HOME=/app/.cache/huggingface
ENV KAGGLE_CACHE=/app/.cache/kaggle

# Create cache directories
RUN mkdir -p /app/.cache/huggingface /app/.cache/kaggle

# Copy custom run_server.py that saves annotated images
COPY main.py /app/run_server.py

EXPOSE 8000

# Run the custom SpeciesNet server with annotation support
# --geofence=False to simplify (no country/region filtering)
CMD ["python", "/app/run_server.py", "--port=8000", "--timeout=120", "--geofence=False", "--save_annotated=True"]
