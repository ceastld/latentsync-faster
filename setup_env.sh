#!/bin/bash
# cudnn9,cuda12.4

# Create a new conda environment
conda create -n latentsync python=3.12.9 -y

conda init

conda activate latentsync

# Install ffmpeg
conda install -y -c conda-forge ffmpeg

# pip install numpy==2.2.3
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install numba

# Python dependencies
pip install -r requirements.txt

# OpenCV dependencies
sudo apt -y install libgl1

# Download all the checkpoints and testdata from HuggingFace
git clone https://huggingface.co/Pinch-Research/latentsync checkpoints
git clone https://huggingface.co/Pinch-Research/latentsync_testset testset


# Soft links for the auxiliary models
mkdir -p ~/.cache/torch/hub/checkpoints
ln -s $(pwd)/checkpoints/auxiliary/2DFAN4-cd938726ad.zip ~/.cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip
ln -s $(pwd)/checkpoints/auxiliary/s3fd-619a316812.pth ~/.cache/torch/hub/checkpoints/s3fd-619a316812.pth
ln -s $(pwd)/checkpoints/auxiliary/vgg16-397923af.pth ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth


# download vae model
python -c "from diffusers import AutoencoderTiny; vae = AutoencoderTiny.from_pretrained('madebyollin/taesd')"