import argparse
import warnings
from latentsync.configs.config import GLOBAL_CONFIG
from latentsync.inference.context import LipsyncContext_v15, LipsyncContext
from latentsync.inference.utils import create_pipeline
from latentsync.inference.utils import set_seed
from latentsync.utils.timer import Timer


def main(args):
    print(f"Input video path: {args.video_path}")
    print(f"Input audio path: {args.audio_path}")

    context = LipsyncContext_v15() if args.v15 else LipsyncContext()
    pipeline = create_pipeline(context)

    set_seed(args.seed)

    pipeline(
        video_path=args.video_path,
        audio_path=args.audio_path,
        video_out_path=args.video_out_path,
    )

    Timer.summary()

    print(f"Output video path: {args.video_out_path}")


if __name__ == "__main__":
    demo = GLOBAL_CONFIG.inference.obama
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default=demo.video_path)
    parser.add_argument("--audio_path", type=str, default=demo.audio_path)
    parser.add_argument("--video_out_path", type=str, default=demo.video_out_path)
    parser.add_argument("--v15", action="store_true")
    parser.add_argument("--seed", type=int, default=1247)
    args = parser.parse_args()

    Timer.enable()

    main(args)
