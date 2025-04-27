import asyncio

import numpy as np
from latentsync import LatentSync
from latentsync.inference.utils import load_audio_clips
from latentsync.utils.video import video_stream
from latentsync.utils.log import setup_logging

setup_logging()

# if use vad, the audio feature for silence will be empty
# then the lipsync model will not process the silence frames
# the results of silence frames will be the original frames

async def main():
    model = LatentSync(use_vad=True, enable_progress=True)
    video = video_stream("assets/obama.mp4")
    audio_clips = load_audio_clips("assets/cxk.mp3", model.context.samples_per_frame)
    for i, frame in enumerate(video):
        model.push_frame(frame)
        audio = audio_clips[i] if i < len(audio_clips) else np.zeros(640, dtype=np.float32)
        model.push_audio(audio)
    model.add_end_task()
    await model.save_to_video("output/silence_audio.mp4")

if __name__ == "__main__":
    asyncio.run(main())
