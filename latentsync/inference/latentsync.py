import argparse
import asyncio
from functools import cached_property
from typing import List, Set, Union

import numpy as np
from tqdm import tqdm
from latentsync.configs.config import GLOBAL_CONFIG
from latentsync.inference.lipsync_infer import LipsyncInference, LipsyncRestore
from latentsync.inference.audio_infer import AudioInference
from latentsync.inference.context import LipsyncContext, LipsyncContext_v15
from latentsync.inference.face_infer import FaceInference
from latentsync.inference.multi_infer import MultiThreadInference
from latentsync.inference.utils import load_audio_clips
from latentsync.pipelines.metadata import LipsyncMetadata
from latentsync.utils.affine_transform import AlignRestore
from latentsync.utils.timer import Timer
from latentsync.utils.video import cycle_video_stream, VideoReader, save_frames_to_video


class LatentSyncInference:
    def __init__(self, context: LipsyncContext, worker_timeout=60, enable_progress=False):
        self.context = context
        self.enable_progress = enable_progress

        self.loop = asyncio.get_event_loop()
        self.tasks: Set[asyncio.Task] = set()
        self.metadata_queue: asyncio.Queue[LipsyncMetadata] = asyncio.Queue()
        self.audio_feature_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()

        self.lipsync_model = LipsyncInference(context=context, worker_timeout=worker_timeout)
        self.lipsync_model.start_workers()
        self.lipsync_model.wait_worker_loaded()
        self.face_model = FaceInference(context=context, worker_timeout=worker_timeout)
        self.face_model.start_workers()
        self.audio_model = AudioInference(context=context, worker_timeout=worker_timeout)
        self.audio_model.start_workers()
        self.lipsync_restore = LipsyncRestore(context=context, worker_timeout=worker_timeout)
        self.lipsync_restore.start_workers()
        self.stopped = False
        self.wait_loaded()

    @property
    def workers(self):
        for k, v in self.__dict__.items():
            if isinstance(v, MultiThreadInference):
                yield v

    def wait_loaded(self):
        for worker in self.workers:
            worker.wait_worker_loaded()

    def push_face(self, frame: np.ndarray):
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
        self.create_task(self.process_frame())
        self.create_task(self.process_audio())
        self.create_task(self.push_data_to_lipsync())
        self.create_task(self.push_data_to_lipsync_restore())

    async def process_frame(self):
        pbar = None
        async for data in self.face_model.result_stream():
            await self.metadata_queue.put(data)
            if pbar is None:
                pbar = tqdm(desc="Processing face", disable=not self.enable_progress)
            pbar.update(1)
        self.metadata_queue.put_nowait(None)
        pbar.close()

    async def process_audio(self):
        pbar = None
        async for data in self.audio_model.result_stream():
            await self.audio_feature_queue.put(data)
            if pbar is None:
                pbar = tqdm(desc="Processing audio", disable=not self.enable_progress)
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

    async def push_data_to_lipsync_restore(self):
        pbar = None
        async for metadata in self.lipsync_model.result_stream():
            self.lipsync_restore.push_data(metadata)
            if pbar is None:
                pbar = tqdm(desc="Pushing data to lipsync restore", disable=not self.enable_progress)
            pbar.update(1)
        self.lipsync_restore.add_end_task()
        pbar.close()

    def result_stream(self):
        return self.lipsync_restore.result_stream()

    def add_end_task(self):
        for worker in (self.audio_model, self.face_model):
            worker.add_end_task()


class LatentSync:
    def __init__(self, version=None, enable_progress=False, video_fps: int = 25, worker_timeout: int = 60):
        self.context = LipsyncContext.from_version(version)
        self.enable_progress = enable_progress
        self.video_fps = video_fps
        self.model = LatentSyncInference(
            context=self.context,
            enable_progress=enable_progress,
            worker_timeout=worker_timeout,
        )
        self.model.start_processing()

    def stop_workers(self):
        self.model.stop_workers()

    async def test(
        self,
        video_path,
        audio_path,
        save_path=None,
        max_frames: int = 200,
    ):
        self.auto_push_data(video_path, audio_path, max_frames, fps=self.video_fps)
        save = save_path is not None
        results = await self.get_all_results(max_frames, save)
        if save:
            save_frames_to_video(results, save_path, audio_path=audio_path)
            print(f"Saved to {save_path}")
        self.stop_workers()

    def push_frames(self, frame: Union[np.ndarray, List[np.ndarray]]):
        if isinstance(frame, np.ndarray):
            self.model.push_face(frame)
        elif isinstance(frame, list):
            for f in frame:
                self.model.push_face(f)
        else:
            raise ValueError(f"Invalid frame type: {type(frame)}")

    def push_audio(self, audio: np.ndarray):
        """
        audio: np.ndarray, sample_rate: 16000
        """
        spf = self.context.samples_per_frame
        if len(audio) % spf != 0:
            audio = np.pad(audio, (0, spf - len(audio) % spf), mode="constant")
        self.model.push_audio(audio)

    def auto_push_data(self, video_path, audio_path, max_frames: int = None, fps: int = 30):
        self.model.create_task(self._auto_push_data(video_path, audio_path, max_frames, fps))

    async def _auto_push_data(self, video_path, audio_path, max_frames: int = None, fps: int = 30):
        audio_clips = load_audio_clips(audio_path, self.context.samples_per_frame)
        frame_interval = 1 / fps  # Target frame interval for 30fps
        last_frame_time = asyncio.get_event_loop().time()

        for i, frame in enumerate(cycle_video_stream(video_path, max_frames=max_frames)):
            current_time = asyncio.get_event_loop().time()
            elapsed = current_time - last_frame_time

            # Calculate sleep time to maintain 30fps
            sleep_time = max(0, frame_interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

            self.push_frames(frame)
            self.push_audio(audio_clips[i % len(audio_clips)])
            last_frame_time = asyncio.get_event_loop().time()

        self.model.add_end_task()

    def result_stream(self):
        return self.model.result_stream()

    async def get_all_results(self, total: int = None, save: bool = False):
        pbar = None
        output_frames = []
        async for data in self.model.result_stream():
            if save:
                output_frames.append(data)
            if pbar is None:
                pbar = tqdm(desc="results", total=total, disable=not self.enable_progress)
            pbar.update(1)
        pbar.close()
        return output_frames
