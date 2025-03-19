# LatentSync

<div align="center">

![LatentSync Logo](https://img.shields.io/badge/LatentSync-AI%20Synchronization-blue)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Pinch--Research%2Flatentsync-yellow)](https://huggingface.co/Pinch-Research/latentsync)

</div>

## 🚀 快速开始

### 环境配置

```bash
# 配置环境（推荐方式）
source setup_env.sh
```

## 🐳 Docker 部署

项目提供两种 Docker 部署方式：

### 方式一：使用 Docker Compose（推荐）

```bash
# 安装必要工具
sudo apt-get install -y nvidia-container-toolkit
sudo apt install docker-compose-plugin

# 构建并启动容器
docker compose up -d

# 进入容器
docker compose exec latentsync bash

# 停止容器
docker compose down
```

### 方式二：使用传统 Docker 命令

```bash
# 构建镜像
sudo apt-get install -y nvidia-container-toolkit
docker build -t latentsync .

# 运行容器
docker run -it --gpus all -v $(pwd):/app -w /app latentsync
```

## 📦 模型资源

### 人脸检测相关模型

本项目使用以下人脸检测模型：

- **人脸检测模型**: [version-RFB-320.onnx](https://github.com/cunjian/pytorch_face_landmark/raw/master/models/onnx/version-RFB-320.onnx)
- **人脸关键点检测模型**: [landmark_detection_56_se_external.onnx](https://github.com/cunjian/pytorch_face_landmark/raw/master/onnx/landmark_detection_56_se_external.onnx)

模型来源: [cunjian/pytorch_face_landmark](https://github.com/cunjian/pytorch_face_landmark)

### 模型库

- **PyTorch Docker 镜像**: [pytorch/pytorch Tags](https://hub.docker.com/r/pytorch/pytorch/tags)
- **HuggingFace 模型**: [Pinch-Research/latentsync](https://huggingface.co/Pinch-Research/latentsync)
