import argparse
import warnings
import torch
from latentsync.inference.context import LipsyncContext
from latentsync.inference.lipsync_model import get_lipsync_pipeline
from accelerate.utils import set_seed as acc_seed
from latentsync.utils.timer import Timer


def set_seed(seed: int):
    if seed != -1:
        acc_seed(seed)
    else:
        torch.seed()
        print(f"Initial seed: {torch.initial_seed()}")

def main(args):
    print(f"Input video path: {args.video_path}")
    print(f"Input audio path: {args.audio_path}")

    context = LipsyncContext()
    pipeline = get_lipsync_pipeline(context)

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
    warnings.filterwarnings("ignore", message="W:onnxruntime")

    # Timer.enable()

    main(args)
