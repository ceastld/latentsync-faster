#!/bin/bash
# ml CUDA/12.4.0

python -m inference \
    --video_path "assets/obama.mp4" \
    --audio_path "assets/cxk.mp3" \
    --video_out_path "output/obama_cxk.mp4"

# python -m inference \
#     --video_path "assets/demo1_video.mp4" \
#     --audio_path "assets/demo1_audio.wav" \
#     --video_out_path "output/demo1_out.mp4"