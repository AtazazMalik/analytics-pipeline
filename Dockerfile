# ============================================================
#  Person Analytics Pipeline — Jetson Nano
#  Base : NVIDIA L4T ML (JetPack 4.6.1 / CUDA 10.2 / TRT 8)
#  Python: 3.6  |  TensorRT: built-in  |  Swap-aware
# ============================================================
FROM nvcr.io/nvidia/l4t-ml:r32.7.1-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV CUDA_MODULE_LOADING=LAZY

# ── System packages ──────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev \
    libhdf5-serial-dev hdf5-tools libhdf5-dev \
    libopenblas-base libopenmpi-dev \
    ffmpeg libavcodec-extra \
    cmake wget curl \
    && rm -rf /var/lib/apt/lists/*

# ── pip / setuptools (Python 3.6 max safe versions) ─────────
RUN pip3 install --upgrade \
    "pip==21.3.1" "setuptools==59.6.0" "wheel==0.37.1"

# ── Core numerics ────────────────────────────────────────────
RUN pip3 install \
    "numpy==1.19.4" \
    "scipy==1.5.4" \
    "pandas==1.1.5" \
    "Pillow==8.4.0" \
    "matplotlib==3.3.4"

# ── PyTorch for Jetson Nano JetPack 4.6.1 + Python 3.6 ──────
RUN pip3 install --no-cache \
    https://developer.download.nvidia.com/compute/redist/jp/v461/pytorch/torch-1.11.0a0+17540c5-cp36-cp36m-linux_aarch64.whl

RUN pip3 install "torchvision==0.12.0"

# ── YOLOv8 (last version supporting Python 3.6) ──────────────
RUN pip3 install "ultralytics==8.0.20"

# ── Verify TensorRT bindings (pre-installed in base image) ───
RUN python3 -c "import tensorrt; print('TensorRT OK:', tensorrt.__version__)"

# ── TensorFlow for Jetson (with CPU fallback) ────────────────
RUN pip3 install --pre \
    --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v461 \
    tensorflow \
    || pip3 install "tensorflow-cpu==2.7.0"

# ── DeepFace + MTCNN (Python 3.6 safe, no retina-face) ───────
RUN pip3 install "deepface==0.0.75" "mtcnn==0.1.1"

# ── App directory structure ───────────────────────────────────
WORKDIR /app
RUN mkdir -p /app/models /app/outputs /app/stage1_output

# ── Copy all pipeline scripts ─────────────────────────────────
COPY scripts/convert_trt.py  /app/convert_trt.py
COPY scripts/stage1.py       /app/stage1.py
COPY scripts/stage2.py       /app/stage2.py
COPY scripts/stage3.py       /app/stage3.py
COPY scripts/run_pipeline.sh /app/run_pipeline.sh

RUN chmod +x /app/run_pipeline.sh

ENTRYPOINT ["/bin/bash", "/app/run_pipeline.sh"]
