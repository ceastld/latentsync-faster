from latentsync import *
import argparse

def main():
    parser = argparse.ArgumentParser(description="LatentSync 视频唇形同步")
    parser.add_argument("--onnx", action="store_true", help="使用ONNX模型加速")
    parser.add_argument("--trt", action="store_true", help="使用TensorRT模型加速")
    parser.add_argument("--time", action="store_true", help="使用时间统计")
    args = parser.parse_args()
    model_type = "ONNX" if args.onnx else "TensorRT" if args.trt else "PyTorch"
    print(f"使用{model_type}模型进行推理...")

    if args.time:
        Timer.enable()

    demo = GLOBAL_CONFIG.inference.obama
    context = LipsyncContext(use_onnx=args.onnx, use_trt=args.trt)
    model = LipsyncModel(context)
    model.inference(
        demo.video_path,
        demo.audio_path,
        demo.video_out_path,
    )
    Timer.summary()
    print(f"输出视频保存到: {demo.video_out_path}")

if __name__ == "__main__":
    main()
