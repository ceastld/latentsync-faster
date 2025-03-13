import argparse
import warnings

import torch
from configs.config import GLOBAL_CONFIG
from latentsync.inference.lipsync_model import get_lipsync_pipeline
from accelerate.utils import set_seed


def main(args):
    print(f"Input video path: {args.video_path}")
    print(f"Input audio path: {args.audio_path}")
    is_fp16_supported = (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    )
    dtype = torch.float16 if is_fp16_supported else torch.float32

    pipeline = get_lipsync_pipeline(dtype, "cuda", use_compile=False)
    config = GLOBAL_CONFIG.unet_config

    pipeline(
        video_path=args.video_path,
        audio_path=args.audio_path,
        video_out_path=args.video_out_path,
        video_mask_path=args.video_out_path.replace(".mp4", "_mask.mp4"),
        num_frames=config.data.num_frames,
        num_inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        weight_dtype=dtype,
        width=config.data.resolution,
        height=config.data.resolution,
    )
    
    if args.seed != -1:
        set_seed(args.seed)
    else:
        torch.seed()

        print(f"Initial seed: {torch.initial_seed()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=3)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=1247)
    args = parser.parse_args()
    warnings.filterwarnings("ignore", message="Initializer .*")

    main(args)

