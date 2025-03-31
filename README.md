# LatentSync

<div align="center">

![LatentSync Logo](https://img.shields.io/badge/LatentSync-AI%20Synchronization-blue)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Pinch--Research%2Flatentsync-yellow)](https://huggingface.co/Pinch-Research/latentsync)

</div>

## 📝 项目简介

LatentSync 是一个先进的唇形同步（Lip Sync）AI 工具，能够将输入视频中的人脸口型与目标音频进行智能同步。通过深度学习技术，系统可以自动分析音频内容，生成与之匹配的口型动作，实现自然的音视频同步效果。

## 🚀 快速开始

### 环境配置

```bash
# 配置环境（推荐方式）
source setup_env.sh
```

### 文档

- 详细的 Docker 部署说明请参考： [Docker 部署指南](doc/docker.md)
- 并行推理方法请查看： [使用指南](doc/usage.md)

## 📦 模型资源

### 人脸检测相关模型

本项目使用以下人脸检测模型：

- **人脸检测模型**: [version-RFB-320.onnx](https://github.com/cunjian/pytorch_face_landmark/raw/master/models/onnx/version-RFB-320.onnx)
- **人脸关键点检测模型**: [landmark_detection_56_se_external.onnx](https://github.com/cunjian/pytorch_face_landmark/raw/master/onnx/landmark_detection_56_se_external.onnx)

模型来源: [cunjian/pytorch_face_landmark](https://github.com/cunjian/pytorch_face_landmark)

### 模型库

- **PyTorch Docker 镜像**: [pytorch/pytorch Tags](https://hub.docker.com/r/pytorch/pytorch/tags)
- **HuggingFace 模型**: [Pinch-Research/latentsync](https://huggingface.co/Pinch-Research/latentsync)
