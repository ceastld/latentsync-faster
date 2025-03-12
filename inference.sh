#!/bin/bash
ml CUDA/12.4.0

python -m inference \
    --video_path "assets/obama1.mp4" \
    --audio_path "assets/cxk.mp3" \
    --video_out_path "video_out.mp4"
