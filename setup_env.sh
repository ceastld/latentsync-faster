#!/bin/bash
# cudnn9,cuda12.4

# Get environment name from first argument, default to 'latentsync'
ENV_NAME=${1:-latentsync}

# Create a new conda environment
conda create -n ${ENV_NAME} python=3.12.9 -y

# conda init

conda activate ${ENV_NAME}

# Install ffmpeg
conda install -y -c conda-forge ffmpeg

# pip install numpy==2.2.3
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Python dependencies
# pip install -r requirements.txt
pip install -e .

# OpenCV dependencies
sudo apt -y install libgl1

# Download all the checkpoints and testdata from HuggingFace
if [ ! -d "checkpoints" ]; then
    git clone https://huggingface.co/Pinch-Research/latentsync checkpoints
fi

# if [ ! -d "testset" ]; then
#     git clone https://huggingface.co/datasets/Pinch-Research/latentsync_testset testset
# fi

# Note: Soft links for auxiliary models are now handled in setup.py

# download vae model
python -c "from diffusers import AutoencoderTiny; vae = AutoencoderTiny.from_pretrained('madebyollin/taesd')"
python -c "from diffusers import AutoencoderKL; vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')"
