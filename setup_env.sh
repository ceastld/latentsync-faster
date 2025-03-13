#!/bin/bash

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

# Download all the checkpoints from HuggingFace
huggingface-cli download ByteDance/LatentSync --local-dir checkpoints --exclude "*.git*" "README.md"

# Soft links for the auxiliary models
mkdir -p ~/.cache/torch/hub/checkpoints
ln -s $(pwd)/checkpoints/auxiliary/2DFAN4-cd938726ad.zip ~/.cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip
ln -s $(pwd)/checkpoints/auxiliary/s3fd-619a316812.pth ~/.cache/torch/hub/checkpoints/s3fd-619a316812.pth
ln -s $(pwd)/checkpoints/auxiliary/vgg16-397923af.pth ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth

# 设置人脸检测模型
echo "设置人脸检测模型..."
python scripts/setup_face_detection.py

pip install torch-tensorrt