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

# Copy the rest of the application code
COPY . .

# Install the package in editable mode
RUN pip3 install -e .

# Download VAE models (matching setup_env.sh)
RUN python3 -c "from diffusers import AutoencoderTiny; vae = AutoencoderTiny.from_pretrained('madebyollin/taesd')"
RUN python3 -c "from diffusers import AutoencoderKL; vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')"

# Download checkpoints from HuggingFace
# RUN git clone https://huggingface.co/Pinch-Research/latentsync checkpoints
# Copy the rest of the application
# COPY ./checkpoints ./checkpoints
