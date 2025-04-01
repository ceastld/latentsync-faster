from functools import cached_property
from typing import List, Optional
import numpy as np
import torch
from tqdm import tqdm
from latentsync.inference.audio_infer import AudioProcessor
from latentsync.inference.context import LipsyncContext
from latentsync.inference.utils import align_audio_features, create_diffusion_pipeline, load_audio_clips
from latentsync.pipelines.metadata import LipsyncMetadata
from latentsync.utils.affine_transform import AlignRestore
from latentsync.utils.timer import Timer
from latentsync.utils.video import VideoReader, save_frames_to_video
from latentsync.utils.frame_preprocess import process_frames


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
        synced_faces_batch, _ = self.pipeline._run_diffusion_batch(
            faces,
            audio_features,
            self.context,
        )
        return synced_faces_batch
    
    def process_metadata_batch(self, metadata_list: List[LipsyncMetadata]):
        """
        audio feature is stored in metadata
        """
        audio_features = [metadata.audio_feature_tensor for metadata in metadata_list]
        return self.process_batch(metadata_list, audio_features)

    @Timer()
    @torch.no_grad()
    def process_batch(
        self,
        metadata_list: List[LipsyncMetadata],
        audio_features: Optional[List[torch.Tensor]],
    ) -> List[LipsyncMetadata]:
        faces = torch.stack([metadata.face_tensor for metadata in metadata_list])
        audio_features = align_audio_features(audio_features, len(faces))

        synced_faces_batch = self.run_diffusion(faces, audio_features)

        for i, metadata in enumerate(metadata_list):
            metadata.set_sync_face(synced_faces_batch[i])

        return metadata_list

    @Timer()
    @torch.no_grad()
    def restore_batch(self, metadata_list: List[LipsyncMetadata]):
        output_frames = []
        for metadata in metadata_list:
            result = metadata.original_frame
            if metadata.sync_face is not None:
                result = self.restorer.restore_img(metadata.original_frame, metadata.sync_face, metadata.affine_matrix)
            output_frames.append(result)
        return output_frames

    @property
    def face_processor(self):
        return self.pipeline.face_processor

    @torch.no_grad()
    def process_video(self, video_frames: List[np.ndarray], audio_samples: np.ndarray):
        """Process a video with corresponding audio samples."""
        assert self.context is not None
        audio_features = self.audio_processor.process_audio(audio_samples, len(video_frames))
        batch_size = self.context.num_frames
        for i in range(0, len(video_frames), batch_size):
            batch_frames = video_frames[i : i + batch_size]
            batch_audio_features = audio_features[i : i + batch_size] if audio_features else None
            metadata_list = self.face_processor.prepare_face_batch(batch_frames)
            if not metadata_list:
                for frame in batch_frames:
                    yield frame
                continue
            batch_output_frames = self.process_batch(metadata_list, batch_audio_features)
            for frame in batch_output_frames:
                yield frame

            print(f"Processed batch {i//batch_size + 1}/{(len(video_frames) + batch_size - 1) // batch_size}")


    def inference(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        is_compress: bool = False,
        is_noise: bool = False,
        is_blur: bool = False,
    ):
        context = self.context
        batch_size = context.num_frames

        # Load video
        video_reader = VideoReader(video_path)
        fps = video_reader.fps
        total_frames = video_reader.total_frames

        # Load audio
        audio_clips = load_audio_clips(audio_path, context.samples_per_frame)

        # Limit total frames to available audio
        total_frames = min(total_frames, len(audio_clips))

        # Process video frames
        processed_frames = []
        frame_idx = 0

        # Keep track of last processed audio for context
        last_audio_samples = None

        face_processor = self.face_processor # pre init
        audio_processor = self.audio_processor # pre init

        pbar = tqdm(total=total_frames, desc="Processing frames")

        while frame_idx < total_frames:
            batch_metadata = []
            frames_batch = video_reader.read_batch(min(batch_size, total_frames - frame_idx))
            frames_batch = process_frames(frames_batch, is_compress=is_compress, is_noise=is_noise, is_blur=is_blur)
            batch_metadata = face_processor.prepare_face_batch(frames_batch)
            if not batch_metadata:
                break
            current_audio_samples = audio_clips[frame_idx : frame_idx + len(frames_batch)].flatten()
            batch_audio_features = audio_processor.process_audio_with_pre(last_audio_samples, current_audio_samples)
            last_audio_samples = current_audio_samples
            output_metadata: List[LipsyncMetadata] = self.process_batch(
                metadata_list=batch_metadata,
                audio_features=batch_audio_features,
            )
            output_frames = self.restore_batch(output_metadata)
            processed_frames.extend(output_frames)
            frame_idx += len(frames_batch)
            pbar.update(len(frames_batch))

        pbar.close()
        video_reader.release()

        # Save output video
        if processed_frames:
            save_frames_to_video(processed_frames, output_path, audio_path, fps)
