import asyncio

import numpy as np
from latentsync import LatentSync

async def main():
    model = LatentSync()
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    model.push_frames([image]*25)
    audio = np.zeros(16000, dtype=np.float32)
    model.push_audio(audio)
    model.add_end_task()
    await model.save_to_video("output/custom_stream.mp4")
    
if __name__ == "__main__":
    asyncio.run(main())
