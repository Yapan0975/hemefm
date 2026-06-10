# NVIDIA CUDA 12.6 development image — sufficient for PyTorch 2.6+ cu126 wheels.
# For RTX 5090 (Blackwell, sm_120) substitute the cu128 base when nightly is required:
#   FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        wget \
        python3.11 python3.11-dev python3.11-venv \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install uv (https://docs.astral.sh/uv/) — fast Rust-based Python package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.cargo/bin/uv /usr/local/bin/uv

WORKDIR /workspace/hemefm

# Copy project files in dependency-cache-friendly order
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY tests/ ./tests/

# Build the venv and install dependencies (editable)
RUN uv venv --python 3.11 && \
    . .venv/bin/activate && \
    uv pip install -e ".[dev]"

ENV PATH="/workspace/hemefm/.venv/bin:${PATH}"

# Default: run smoke test
CMD ["python", "-m", "hemefm.train", "experiment=hello_world"]
