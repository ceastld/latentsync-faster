from typing import List, Optional
import numpy as np
import torch

from latentsync.inference.context import LipsyncContext
from latentsync.utils.timer import Timer
from .multi_infer import MultiProcessInference


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


class AudioInfer(MultiProcessInference):
    def __init__(self, num_workers=1, worker_timeout=60):
        super().__init__(num_workers, worker_timeout)

    def get_model(self):
        pass

    def worker(self):
        pass

    def push_audio(self, audio: np.ndarray):
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
