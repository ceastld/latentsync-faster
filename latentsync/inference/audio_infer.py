import asyncio
from typing import List, Optional, Union, Any, Dict
import numpy as np
import torch
from tqdm import tqdm
from latentsync.configs.config import GLOBAL_CONFIG
from latentsync.inference.context import LipsyncContext
from latentsync.inference.utils import align_audio_features, load_audio_clips, preprocess_audio
from latentsync.utils.timer import Timer
from latentsync.whisper.whisper.audio import load_audio
from latentsync.inference.multi_infer import MultiThreadInference, InferenceTask
from latentsync.inference.buffer_infer import BufferInference
from latentsync.pipelines.metadata import AudioMetadata


class AudioProcessor:
    def __init__(self, context: LipsyncContext):
        self.context: LipsyncContext = context
        self.audio_encoder = context.audio_encoder

    @torch.no_grad()
    def process_audio(self, audio_samples: np.ndarray, num_faces: int = -1) -> Optional[torch.Tensor]:
        """Process audio samples and align them with the number of faces."""
        # Process audio samples for this batch
        whisper_feature = self.audio_encoder.samples2feat(audio_samples)

        audio_features = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=self.context.video_fps)

        # Align audio features with the number of faces
        return align_audio_features(audio_features, num_faces)

    # @Timer()
    @torch.no_grad()
    def process_audio_with_pre(self, pre_audio_samples, audio_samples):
        samples_per_frame = self.context.samples_per_frame
        num_frames = int(np.ceil(len(audio_samples) / samples_per_frame))

        if pre_audio_samples is not None:
            combined_samples = np.concatenate([pre_audio_samples, audio_samples])
        else:
            combined_samples = audio_samples

        combined_features = self.process_audio(combined_samples)
        return combined_features[-num_frames:] if num_frames > 0 else []


class AudioInference(MultiThreadInference[np.ndarray, np.ndarray]):
    def __init__(
        self,
        context: LipsyncContext,
        num_workers=1,
        worker_timeout=60,
        enable_timer=False,
    ):
        super().__init__(num_workers, worker_timeout, enable_timer)
        self.context = context

    def get_model(self):
        return AudioProcessor(self.context)

    def worker(self):
        self.audio_buffer = []
        self.result_start_id = 0
        self.last_audio_samples = None
        super().worker()

    def process_task(self, model: AudioProcessor, task: InferenceTask[np.ndarray]) -> None:
        audio_buffer = self.audio_buffer
        if len(audio_buffer) == 0:
            self.result_start_idx = task.idx
        audio_buffer.append(task.data)
        if len(audio_buffer) >= self.context.audio_batch_size:
            audio_samples = np.concatenate(audio_buffer)
            result = model.process_audio_with_pre(self.last_audio_samples, audio_samples)
            for i in range(len(audio_buffer)):
                self._set_result(self.result_start_idx + i, result[i].cpu().numpy())
            self.audio_buffer = []
            self.last_audio_samples = audio_samples

    def push_audio(self, audio: np.ndarray):
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        clips = audio.reshape(-1, self.context.samples_per_frame)
        self.add_tasks(clips)


class AudioBatchInference(BufferInference[np.ndarray, AudioMetadata]):
    def __init__(self, context: LipsyncContext, num_workers=1, worker_timeout=60):
        super().__init__(context.audio_batch_size, num_workers, worker_timeout)
        self.context = context
        self.last_audio_samples = None

    def get_model(self):
        return AudioProcessor(self.context)

    def push_audio(self, audio: np.ndarray):
        audio_clips = preprocess_audio(audio, self.context.samples_per_frame)
        self.push_data_batch(audio_clips)

    def infer_task(self, model: AudioProcessor, data: List[np.ndarray]):
        audio_samples = np.concatenate(data)
        audio_features = model.process_audio_with_pre(self.last_audio_samples, audio_samples)
        self.last_audio_samples = audio_samples
        results = []
        for audio_clip, audio_feature in zip(data, audio_features):
            results.append(
                AudioMetadata(
                    audio_samples=audio_clip,
                    audio_feature=audio_feature.cpu().numpy(),
                )
            )
        return results


async def auto_push_audio(audio_file, infer: AudioInference):
    print("Push Audio Start")
    audio = load_audio(audio_file)
    total_clips = len(audio) // infer.context.samples_per_frame
    audio_clips = audio[: total_clips * infer.context.samples_per_frame].reshape(-1, infer.context.samples_per_frame)
    with tqdm(total=total_clips, desc="Push Audio") as pbar:
        for i in range(total_clips):
            infer.push_audio(audio_clips[i])
            pbar.update(1)
            # await asyncio.sleep(0.04)
    infer.add_end_task()
    return audio_clips


async def wait_for_results(infer: AudioInference):
    results = []
    with tqdm(desc="Processing Audio") as pbar:
        async for result in infer.result_stream():
            if result is None:
                break
            results.append(result)
            pbar.update(1)
    return results


async def main():
    context = LipsyncContext()
    infer = AudioInference(context)
    infer.start_workers()
    infer.wait_worker_loaded()
    await auto_push_audio(GLOBAL_CONFIG.inference.default_audio_path, infer)
    results = await wait_for_results(infer)
    # print(results)
    infer.dispose()
    print("Done")


if __name__ == "__main__":
    # Timer.enable()
    asyncio.run(main())
    Timer.summary()
