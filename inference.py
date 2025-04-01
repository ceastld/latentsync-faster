import argparse
from latentsync import *

def main(args):
    print(f"Input video path: {args.video_path}")
    print(f"Input audio path: {args.audio_path}")
    context = LipsyncContext.from_version(args.version or "v15")
    pipeline = create_pipeline(context)
    pipeline(
        video_path=args.video_path,
        audio_path=args.audio_path,
        video_out_path=args.video_out_path,
    )


if __name__ == "__main__":
    demo = GLOBAL_CONFIG.inference.obama
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default=demo.video_path)
    parser.add_argument("--audio_path", type=str, default=demo.audio_path)
    parser.add_argument("--video_out_path", type=str, default=demo.video_out_path)
    parser.add_argument("--version", type=str, default="v15")
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--time", action="store_true")
    args = parser.parse_args()
    if args.time:
        Timer.enable()
    main(args)
    Timer.summary()
    print(f"Output video path: {args.video_out_path}")

