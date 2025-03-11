#!/bin/bash

python -m inference \
    --unet_config_path "configs/unet/second_stage.yaml" \
    --inference_ckpt_path "checkpoints/latentsync_unet.pt" \
    --inference_steps 3 \
    --guidance_scale 1.5 \
    --video_path "assets/obama.mp4" \
    --audio_path "assets/cxk.mp3" \
    --video_out_path "video_out.mp4"
