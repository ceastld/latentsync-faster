from latentsync import *
import argparse


def main():
    parser = argparse.ArgumentParser(description="LatentSync 视频唇形同步")
    parser.add_argument("--time", action="store_true", help="使用时间统计")
    parser.add_argument("--video_path", type=str, help="视频路径")
    parser.add_argument("--audio_path", type=str, help="音频路径")
    parser.add_argument("--output_path", type=str, help="输出路径")
    args = parser.parse_args()
    print("开始进行推理...")

    if args.time:
        Timer.enable()

    demo = GLOBAL_CONFIG.inference.obama
    context = LipsyncContext.from_version("v15")
    model = LipsyncModel(context)
    if args.video_path and args.audio_path and args.output_path:
        model.inference(args.video_path, args.audio_path, args.output_path)
    else:
        model.inference(demo.video_path, demo.audio_path, demo.video_out_path)
    Timer.summary()
    if args.output_path:
        print(f"输出视频保存到: {args.output_path}")
    else:
        print(f"输出视频保存到: {demo.video_out_path}")


if __name__ == "__main__":
    main()
