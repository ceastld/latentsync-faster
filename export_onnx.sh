#!/bin/bash

# 确保脚本失败时立即退出
set -e

# 定义默认参数
MODEL_PATH=""
OUTPUT_PATH="unet.onnx"
BATCH_SIZE=1
FRAMES=8
HEIGHT=64
WIDTH=64
USE_EXTERNAL_DATA=false
USE_FP16=false

# 打印使用说明
function print_usage {
    echo "用法: $0 --model_path <模型路径> [--output_path <输出路径>] [--batch_size <批次大小>] [--frames <帧数>] [--height <高度>] [--width <宽度>] [--use_external_data] [--fp16]"
    echo ""
    echo "参数:"
    echo "  --model_path        UNet模型权重文件路径 (必需)"
    echo "  --output_path       ONNX输出文件路径 (默认: unet.onnx)"
    echo "  --batch_size        样本批次大小 (默认: 1)"
    echo "  --frames            样本帧数 (默认: 8)"
    echo "  --height            样本高度 (默认: 64)"
    echo "  --width             样本宽度 (默认: 64)"
    echo "  --use_external_data 使用外部数据格式 (默认: 不使用)"
    echo "  --fp16              使用FP16精度量化模型 (默认: 不使用)"
    echo ""
    echo "示例:"
    echo "  $0 --model_path ./checkpoints/unet.pth --output_path ./output/unet.onnx"
    echo "  $0 --model_path ./checkpoints/unet.pth --output_path ./output/unet.onnx --use_external_data"
    echo "  $0 --model_path ./checkpoints/unet.pth --output_path ./output/unet_fp16.onnx --fp16"
    exit 1
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --model_path)
            MODEL_PATH="$2"
            shift
            shift
            ;;
        --output_path)
            OUTPUT_PATH="$2"
            shift
            shift
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift
            shift
            ;;
        --frames)
            FRAMES="$2"
            shift
            shift
            ;;
        --height)
            HEIGHT="$2"
            shift
            shift
            ;;
        --width)
            WIDTH="$2"
            shift
            shift
            ;;
        --use_external_data)
            USE_EXTERNAL_DATA=true
            shift
            ;;
        --fp16)
            USE_FP16=true
            shift
            ;;
        *)
            echo "未知参数: $1"
            print_usage
            ;;
    esac
done

# 检查必需参数
if [[ -z "$MODEL_PATH" ]]; then
    echo "错误: 必须提供模型路径 (--model_path)"
    print_usage
fi

# 准备外部数据参数
EXTERNAL_DATA_ARG=""
if [[ "$USE_EXTERNAL_DATA" == "true" ]]; then
    EXTERNAL_DATA_ARG="--use_external_data"
    echo "将使用外部数据格式导出模型"
fi

# 准备FP16参数
FP16_ARG=""
if [[ "$USE_FP16" == "true" ]]; then
    FP16_ARG="--fp16"
    echo "将模型量化为FP16精度"
fi

# 清理目标目录
OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
if [[ -n "$OUTPUT_DIR" && "$OUTPUT_DIR" != "." ]]; then
    mkdir -p "$OUTPUT_DIR"
fi

# 运行Python导出脚本
echo "正在导出UNet模型为ONNX格式..."
python export_unet_to_onnx.py \
    --model_path "$MODEL_PATH" \
    --output_path "$OUTPUT_PATH" \
    --sample_batch_size "$BATCH_SIZE" \
    --sample_frames "$FRAMES" \
    --sample_height "$HEIGHT" \
    --sample_width "$WIDTH" \
    $EXTERNAL_DATA_ARG \
    $FP16_ARG

echo "导出完成！"
echo "ONNX模型已保存为: $OUTPUT_PATH" 