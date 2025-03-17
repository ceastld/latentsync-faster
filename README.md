03/08/25: The dependencies are very confusing because of the application of the newest `diffusers` and `transformers`. But it is necessary for `torch.compile`. It can be run by installing every newest dependency seperately now. Later I will try to create a stable requirements file.

03/09/25: The correct requirements have been settled down. Just `sh setup_env.sh`

03/11/25: The inference process has been packaged into several functions in batch mode. Search `core function` to find the key components of face_processor, audio_processor (todo) and diffusion_processor.

# Docker
pytorch docker images: [pytorch/pytorch Tags | Docker Hub](https://hub.docker.com/r/pytorch/pytorch/tags)

huggingface model repo: [Pinch-Research/latentsync · Hugging Face](https://huggingface.co/Pinch-Research/latentsync)

## build
```bash
sudo apt-get install -y nvidia-container-toolkit
docker build -t latentsync .
```

## run
```bash
docker run -it --gpus all -v $(pwd):/app -w /app latentsync
```

## 人脸检测模型

项目使用以下人脸检测模型：

- 人脸检测模型: [pytorch_face_landmark/version-RFB-320.onnx](https://github.com/cunjian/pytorch_face_landmark/raw/master/models/onnx/version-RFB-320.onnx)
- 人脸关键点检测模型: [pytorch_face_landmark/landmark_detection_56_se_external.onnx](https://github.com/cunjian/pytorch_face_landmark/raw/master/onnx/landmark_detection_56_se_external.onnx)

这些模型来源于 [cunjian/pytorch_face_landmark](https://github.com/cunjian/pytorch_face_landmark) 项目。
