import asyncio
from functools import cached_property
from typing import Set

import numpy as np
from tqdm import tqdm
from latentsync.configs.config import GLOBAL_CONFIG
from latentsync.inference.lipsync_infer import LipsyncInference
from latentsync.inference.audio_infer import AudioInference
from latentsync.inference.context import LipsyncContext
from latentsync.inference.face_infer import FaceInference
from latentsync.inference.multi_infer import MultiThreadInference
from latentsync.inference.utils import load_audio_clips
from latentsync.pipelines.metadata import LipsyncMetadata
from latentsync.utils.timer import Timer
from latentsync.utils.video import cycle_video_stream, VideoReader


class LatentSyncInference:
    def __init__(self, context: LipsyncContext, worker_timeout=60):
        self.context = context
        self.loop = asyncio.get_event_loop()
        self.tasks: Set[asyncio.Task] = set()
        self.metadata_queue: asyncio.Queue[LipsyncMetadata] = asyncio.Queue()
        self.audio_feature_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()

        self.lipsync_model = LipsyncInference(context=context, worker_timeout=worker_timeout)
        self.lipsync_model.start_workers()
        self.face_model = FaceInference(context=context, worker_timeout=worker_timeout)
        self.face_model.start_workers()
        self.audio_model = AudioInference(context=context, worker_timeout=worker_timeout)
        self.audio_model.start_workers()
        self.stopped = False

    @property
    def workers(self):
        for k, v in self.__dict__.items():
            if isinstance(v, MultiThreadInference):
                yield v

    @Timer()
    def wait_loaded(self):
        for worker in self.workers:
            worker.wait_worker_loaded()

    def push_frame(self, frame: np.ndarray):
        self.face_model.push_frame(frame)

    def push_audio(self, audio: np.ndarray):
        self.audio_model.push_audio(audio)

    def stop_workers(self):
        if self.stopped:
            return
        for task in self.tasks:
            task.cancel()
        for worker in self.workers:
            worker.stop_workers()
        self.stopped = True

    def create_task(self, coro):
        task = self.loop.create_task(coro)
        self.tasks.add(task)

    def is_alive(self):
        return (
            any(worker.is_alive() for worker in self.workers)
            or not self.metadata_queue.empty()
            or not self.audio_feature_queue.empty()
        )

    def start_processing(self):
        # self.create_task(self.)
        self.create_task(self.process_face())
        self.create_task(self.process_audio())
        self.create_task(self.push_data_to_lipsync())

    async def process_face(self):
        pbar = tqdm(desc="Processing face")
        async for data in self.face_model.result_stream():
            await self.metadata_queue.put(data)
            pbar.update(1)
        self.metadata_queue.put_nowait(None)
        pbar.close()

    async def process_audio(self):
        pbar = tqdm(desc="Processing audio")
        async for data in self.audio_model.result_stream():
            await self.audio_feature_queue.put(data)
            pbar.update(1)
        self.audio_feature_queue.put_nowait(None)
        pbar.close()

    async def push_data_to_lipsync(self):
        while self.is_alive():
            metadata = await self.metadata_queue.get()
            audio_feature = await self.audio_feature_queue.get()
            if metadata is None or audio_feature is None:
                self.lipsync_model.add_end_task()
                break
            metadata.audio_feature = audio_feature
            self.lipsync_model.push_data(metadata)

    async def wait_for_results(self):
        async for data in self.lipsync_model.result_stream():
            yield data

    def add_end_task(self):
        for worker in (self.audio_model, self.face_model):
            worker.add_end_task()


async def auto_push_data(video_path, audio_path, model: LatentSyncInference, max_frames: int = 200):
    audio_clips = load_audio_clips(audio_path, model.context.samples_per_frame)
    for i, frame in enumerate(cycle_video_stream(video_path, max_frames)):
        model.push_frame(frame)
        model.push_audio(audio_clips[i % len(audio_clips)])
        await asyncio.sleep(1 / 25)
    model.add_end_task()


async def wait_for_results(model: LatentSyncInference):
    pbar = tqdm(desc="results")
    async for data in model.wait_for_results():
        pbar.update(1)
    pbar.close()


async def main():
    context = LipsyncContext()
    model = LatentSyncInference(context)
    model.wait_loaded()
    model.start_processing()
    asyncio.create_task(
        auto_push_data(
            GLOBAL_CONFIG.inference.default_video_path,
            GLOBAL_CONFIG.inference.default_audio_path,
            model,
        )
    )
    await wait_for_results(model)

    model.stop_workers()


if __name__ == "__main__":
    # Timer.enable()
    asyncio.run(main())
    Timer.summary()
