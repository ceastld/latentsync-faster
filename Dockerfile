FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    python3-pip \
    python3-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and setup files
COPY requirements.txt .
COPY scripts/setup_face_detection.py scripts/

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install --no-cache-dir numba
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir huggingface_hub

# Create directories for checkpoints
RUN mkdir -p checkpoints/whisper checkpoints/auxiliary
RUN mkdir -p ~/.cache/torch/hub/checkpoints

# Download checkpoints from HuggingFace
RUN python3 -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='ByteDance/LatentSync', filename='latentsync_unet.pt', local_dir='checkpoints'); \
    hf_hub_download(repo_id='ByteDance/LatentSync', filename='whisper/tiny.pt', local_dir='checkpoints')"

# Set up face detection models
RUN python3 scripts/setup_face_detection.py

# Copy the rest of the application
COPY . .

# Create symbolic links for auxiliary models
RUN ln -sf $(pwd)/checkpoints/auxiliary/2DFAN4-cd938726ad.zip ~/.cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip || true
RUN ln -sf $(pwd)/checkpoints/auxiliary/s3fd-619a316812.pth ~/.cache/torch/hub/checkpoints/s3fd-619a316812.pth || true
RUN ln -sf $(pwd)/checkpoints/auxiliary/vgg16-397923af.pth ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth || true

# Make inference script executable
RUN chmod +x inference.sh

# Set default command
# ENTRYPOINT ["./inference.sh"] 