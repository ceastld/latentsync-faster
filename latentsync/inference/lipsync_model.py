from functools import cached_property
from typing import List, Optional
import numpy as np
import torch
from latentsync.inference.audio_infer import AudioProcessor
from latentsync.inference.context import LipsyncContext
from latentsync.inference.utils import create_diffusion_pipeline
from latentsync.pipelines.metadata import LipsyncMetadata
from latentsync.utils.affine_transform import AlignRestore
from latentsync.utils.timer import Timer


class LipsyncModel:
    def __init__(self, context: LipsyncContext):
        self.context: LipsyncContext = context
        self.pipeline = create_diffusion_pipeline(context)
        # Initialize restorer during initialization to avoid first-call delay
        self.restorer = AlignRestore()

    @property
    def device(self):
        return self.context.device

    @cached_property
    def audio_processor(self):
        return AudioProcessor(self.context)

    @torch.no_grad()
    def run_diffusion(self, faces: torch.Tensor, audio_features: Optional[List[torch.Tensor]]) -> torch.Tensor:
        synced_faces_batch, _ = self.pipeline._run_diffusion_batch(faces, audio_features, self.context)
        return synced_faces_batch

    @Timer()
    @torch.no_grad()
    def process_batch(
        self,
        metadata_list: List[LipsyncMetadata],
        audio_features: Optional[List[torch.Tensor]],
    ):
        faces = torch.stack([metadata.face for metadata in metadata_list])
        audio_features = self.audio_processor.align_audio_features(audio_features, len(faces))

        synced_faces_batch = self.run_diffusion(faces, audio_features)

        for i, metadata in enumerate(metadata_list):
            metadata.set_sync_face(synced_faces_batch[i])

        return metadata_list

        # output_frames = [
        #     self.restorer.restore_img(metadata.original_frame, metadata.sync_face, metadata.affine_matrix)
        #     for metadata in metadata_list
        # ]
        # return output_frames

    @Timer()
    @torch.no_grad()
    def restore_batch(self, metadata_list: List[LipsyncMetadata]):
        return [
            self.restorer.restore_img(metadata.original_frame, metadata.sync_face, metadata.affine_matrix)
            for metadata in metadata_list
        ]

    @property
    def face_processor(self):
        return self.pipeline.face_processor

    @torch.no_grad()
    def process_video(self, video_frames: List[np.ndarray], audio_samples: np.ndarray):
        """Process a video with corresponding audio samples."""
        assert self.context is not None

        # 1. Process audio samples
        audio_features = self.audio_processor.process_audio(audio_samples, len(video_frames))

        # 2. Define batch size based on context
        batch_size = self.context.num_frames

        # 3. Process video frames in batches
        for i in range(0, len(video_frames), batch_size):
            # Get current batch of frames
            batch_frames = video_frames[i : i + batch_size]

            # Get corresponding audio features for this batch
            batch_audio_features = audio_features[i : i + batch_size] if audio_features else None

            # Preprocess frames to get metadata
            metadata_list = self.face_processor.prepare_face_batch(batch_frames)

            # Skip batch if no faces were detected
            if not metadata_list:
                for frame in batch_frames:
                    yield frame
                continue

            # Process the batch
            batch_output_frames = self.process_batch(metadata_list, batch_audio_features)

            # Add processed frames to output
            for frame in batch_output_frames:
                yield frame

            print(f"Processed batch {i//batch_size + 1}/{(len(video_frames) + batch_size - 1) // batch_size}")
