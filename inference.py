import argparse
import warnings
from latentsync.inference.context import LipsyncContext
from latentsync.inference.utils import create_pipeline
from latentsync.inference.utils import set_seed
from latentsync.utils.timer import Timer


def main(args):
    print(f"Input video path: {args.video_path}")
    print(f"Input audio path: {args.audio_path}")

    context = LipsyncContext()
    pipeline = create_pipeline(context)

    set_seed(args.seed)

    pipeline(
        video_path=args.video_path,
        audio_path=args.audio_path,
        video_out_path=args.video_out_path,
    )

    print(f"Output video path: {args.video_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1247)
    args = parser.parse_args()

    Timer.enable()

    main(args)
