import asyncio

import numpy as np
from latentsync import LatentSync


async def main():
    model = LatentSync()
    num_frames = 50
    image = np.random.randint(0, 255, (num_frames, 640, 640, 3), dtype=np.uint8)
    model.push_frame([i for i in image])
    audio = np.zeros(640 * num_frames, dtype=np.float32)
    model.push_audio(audio)
    model.add_end_task()
    await model.save_to_video("output/custom_stream.mp4")


if __name__ == "__main__":
    asyncio.run(main())
