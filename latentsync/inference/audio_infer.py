import asyncio
from typing import List, Optional, override
import numpy as np
import torch
from tqdm import tqdm

from latentsync.configs.config import GLOBAL_CONFIG
from latentsync.inference.context import LipsyncContext
from latentsync.utils.timer import Timer
from latentsync.whisper.whisper.audio import load_audio
from latentsync.inference.multi_infer import MultiProcessInference, MultiThreadInference


class AudioProcessor:
    def __init__(self, context: LipsyncContext):
        self.context: LipsyncContext = context
        self.audio_encoder = context.create_audio_encoder()

    @torch.no_grad()
    def process_audio(self, audio_samples: np.ndarray, num_faces: int = -1) -> Optional[torch.Tensor]:
        """Process audio samples and align them with the number of faces."""
        # Process audio samples for this batch
        whisper_feature = self.audio_encoder.samples2feat(audio_samples)

        audio_features = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=self.context.video_fps)

        # Align audio features with the number of faces
        return self.align_audio_features(audio_features, num_faces)

    @Timer()
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

    @torch.no_grad()
    def align_audio_features(self, audio_features: List[torch.Tensor], num_faces: int) -> List[torch.Tensor]:
        """Ensure audio features match the number of faces by padding or trimming."""
        if len(audio_features) < num_faces:
            # Pad with last feature if needed
            last_feature = audio_features[-1] if audio_features else torch.zeros_like(audio_features[0])
            audio_features.extend([last_feature] * (num_faces - len(audio_features)))
        elif num_faces > 0:
            # Trim extra features
            audio_features = audio_features[:num_faces]

        return audio_features


class AudioInference(MultiThreadInference):
    def __init__(
        self,
        context: LipsyncContext,
        num_workers=1,
        worker_timeout=60,
        enable_timer=False,
    ):
        super().__init__(num_workers, worker_timeout, enable_timer)
        self.context = context

    @Timer("audio_infer_model_init")
    def get_model(self):
        return AudioProcessor(self.context)

    @override
    def worker(self):
        self.audio_buffer = []
        self.result_start_id = 0
        self.last_audio_samples = None
        super().worker()

    @override
    def process_task(self, model: AudioProcessor, idx, audio_clip: np.ndarray):
        audio_buffer = self.audio_buffer
        if len(audio_buffer) == 0:
            self.result_start_idx = idx
        audio_buffer.append(audio_clip)
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
