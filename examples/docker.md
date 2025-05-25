# Docker Deployment Guide

This project provides two methods for Docker deployment:

## Method 1: Using Docker Compose (Recommended)

# see [Installing the NVIDIA Container Toolkit — NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#setting-up-nvidia-container-toolkit) to setup the docker environment.

```bash
# Install required tools
sudo apt-get install -y nvidia-container-toolkit
sudo apt install docker-compose-plugin

# Build and start containers
docker compose up -d

# Enter container
docker compose exec latentsync bash

# Stop containers
docker compose down
```

## Method 2: Using Traditional Docker Commands

```bash
# Build image
sudo apt-get install -y nvidia-container-toolkit
docker build -t latentsync .

# Run container
docker run -it --gpus all -v $(pwd):/app -w /app latentsync
```

## Docker Resources

### PyTorch Docker Image
- [pytorch/pytorch Tags](https://hub.docker.com/r/pytorch/pytorch/tags) 