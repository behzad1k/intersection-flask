# Vehicle Counter Web Application - CPU-Only Dockerfile
# For servers without GPU support

FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch CPU version
RUN pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu

# Install MMCV (CPU version)
RUN pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html

# Install MMEngine
RUN pip install mmengine==0.10.7

# Install MMDetection
RUN pip install mmdet==3.3.0

# Install MMYolo
RUN pip install mmyolo==0.6.0

# Install other Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Final production image
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create application user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/uploads /app/outputs /app/weights /app/configs && \
    chown -R appuser:appuser /app

WORKDIR /app

# Copy application files
COPY --chown=appuser:appuser app.py counter_worker.py directional_counter.py ./
COPY --chown=appuser:appuser templates ./templates/
COPY --chown=appuser:appuser static ./static/
COPY --chown=appuser:appuser configs ./configs/

USER appuser

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8002/', timeout=5)" || exit 1

CMD ["python3", "app.py"]