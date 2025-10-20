#!/bin/bash

# Set offline mode environment variables
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Disable network checks
export GRADIO_SERVER_NAME=127.0.0.1
export GRADIO_SERVER_PORT=7860

echo "Starting LatentSync Demo in offline mode..."
echo "Environment variables set:"
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
echo "HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE"

# Run the demo
python demo.py
