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

### Docker 使用

#### 方式一：使用 Docker Compose（推荐）

```bash
# 构建并启动服务
docker-compose up --build

# 后台运行
docker-compose up -d --build

# 停止服务
docker-compose down
```

#### 方式二：使用 Docker 命令

```bash
# 构建镜像
docker build -t latentsync .

# 运行容器
docker run -it --gpus all -p 7860:7860 latentsync

# 后台运行
docker run -d --gpus all -p 7860:7860 --name latentsync-container latentsync
```

#### Docker 环境说明

- **GPU 支持**：容器支持 NVIDIA GPU 加速，需要安装 nvidia-docker2
- **端口映射**：默认映射 7860 端口到主机
- **数据持久化**：可以通过挂载卷来持久化模型和输出文件
- **环境变量**：支持通过环境变量配置 CUDA 设备等参数

#### 高级用法

```bash
# 挂载本地目录到容器
docker run -it --gpus all -p 7860:7860 \
  -v /path/to/your/models:/app/models \
  -v /path/to/your/output:/app/output \
  latentsync

# 指定 CUDA 设备
docker run -it --gpus '"device=0"' -p 7860:7860 latentsync
```



