FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    gnupg \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get -y update
RUN apt-get install -y ffmpeg

# Set working directory
WORKDIR /app

# Copy requirements and setup files
COPY requirements.txt .

# Install Python dependencies in the specified order
# RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel
RUN pip3 install --no-cache-dir numba
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir huggingface_hub

# Create directories for checkpoints
RUN mkdir -p ~/.cache/torch/hub/checkpoints
# Download checkpoints from HuggingFace
# RUN git clone https://huggingface.co/Pinch-Research/latentsync checkpoints
# Copy the rest of the application
# COPY ./checkpoints ./checkpoints

# Create symbolic links for auxiliary models
RUN ln -sf $(pwd)/checkpoints/auxiliary/2DFAN4-cd938726ad.zip ~/.cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip
RUN ln -sf $(pwd)/checkpoints/auxiliary/s3fd-619a316812.pth ~/.cache/torch/hub/checkpoints/s3fd-619a316812.pth
RUN ln -sf $(pwd)/checkpoints/auxiliary/vgg16-397923af.pth ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth
