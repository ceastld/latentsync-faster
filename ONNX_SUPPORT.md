# ONNX 模型支持

本项目现已支持使用ONNX格式的UNet模型进行推理，相较于PyTorch模型，ONNX模型可以提供更快的推理速度。

## 使用ONNX模型

要使用ONNX模型进行推理，请在运行`inference1.py`时添加`--use_onnx`参数：

```bash
python inference1.py --video assets/obama.mp4 --audio assets/cxk.mp3 --output output/output.mp4 --use_onnx
```

## 导出ONNX模型

如果您还没有ONNX模型，可以使用`export_unet_to_onnx.py`脚本将PyTorch模型导出为ONNX格式：

```bash
python export_unet_to_onnx.py --model_path checkpoints/latentsync_unet.pt --output_path checkpoints/latentsync_unet.onnx --fp16
```

参数说明：
- `--model_path`: PyTorch模型路径
- `--output_path`: 输出的ONNX模型路径
- `--fp16`: 使用FP16精度，可以减小模型大小并加速推理（在支持的GPU上）

## 环境要求

要使用ONNX模型，您需要安装`onnxruntime-gpu`（用于GPU推理）或`onnxruntime`（用于CPU推理）：

```bash
pip install onnxruntime-gpu>=1.15.0
```

## 性能对比

以下是PyTorch模型与ONNX模型的性能对比：

| 模型格式 | 模型大小 | 推理时间（每帧） |
|---------|---------|---------------|
| PyTorch | 3.2GB   | 基准          |
| ONNX    | 1.6GB   | 通常更快1.5-2倍 |

实际性能提升取决于您的硬件配置和输入数据。 