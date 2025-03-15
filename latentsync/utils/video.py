import asyncio
from copy import deepcopy
import logging
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from typing import Any, Dict, List, Set, Tuple, Union

import cv2
import librosa
import numpy as np
import torch
from tqdm import tqdm

def print_pt(path):
    data = torch.load(path)
    recursive_print(data)

def recursive_print(d, indent=0):
    # 遍历字典的键值对
    for key, value in d.items():
        # 打印当前的键，缩进表示层级
        print(" " * indent + f"Key: {key}", end="")

        # 如果值是字典，递归调用
        if isinstance(value, dict):
            print()
            recursive_print(value, indent + 4)

        # 如果值是数组（numpy 或 torch），打印形状
        elif isinstance(value, (np.ndarray, torch.Tensor)):
            print(f", Shape: {value.shape}")

        # 如果是其他类型，直接打印值
        else:
            print(f", Value: {value}")


def hstack_videos(input1, input2, output):
    cmd = (
        f"ffmpeg -i {input1} -i {input2} -filter_complex hstack=inputs=2 "
        f"-c:v libx264 -crf 23 -preset fast -y -hide_banner -loglevel error {output}"
    )
    assert subprocess.run(shlex.split(cmd)).returncode == 0


def combine_video_and_audio(video_file, audio_file, output, quality=17, copy_audio=True):
    audio_codec = "-c:a copy" if copy_audio else ""
    cmd = (
        f"ffmpeg -i {video_file} -i {audio_file} -c:v libx264 -crf {quality} -pix_fmt yuv420p "
        f"{audio_codec} -fflags +shortest -y -hide_banner -loglevel error {output}"
    )
    assert subprocess.run(shlex.split(cmd)).returncode == 0


def convert_video(video_file, output, quality=17):
    cmd = f"ffmpeg -i {video_file} -c:v libx264 -crf {quality} -pix_fmt yuv420p " f"-fflags +shortest -y -hide_banner -loglevel error {output}"
    assert subprocess.run(shlex.split(cmd)).returncode == 0


def reencode_audio(audio_file, output):
    cmd = f"ffmpeg -i {audio_file} -y -hide_banner -loglevel error {output}"
    assert subprocess.run(shlex.split(cmd)).returncode == 0


def save_frames_to_video(frames, out_path, audio_path=None, fps=25, save_images=False):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_video_file = tempfile.NamedTemporaryFile("w", suffix=".mp4", dir=out_path.parent)
    if save_images:
        out_image_dir = out_path.with_suffix("")
        out_image_dir.mkdir(exist_ok=True)
    with LazyVideoWriter(tmp_video_file.name, fps=fps, save_images=save_images) as writer:
        for frame in frames:
            writer.write(frame)
    if audio_path is not None:
        # needs to re-encode audio to AAC format first, or the audio will be ahead of the video!
        tmp_audio_file = tempfile.NamedTemporaryFile("w", suffix=".mp3", dir=out_path.parent)
        reencode_audio(audio_path, tmp_audio_file.name)
        combine_video_and_audio(tmp_video_file.name, tmp_audio_file.name, out_path)
        tmp_audio_file.close()
    else:
        convert_video(tmp_video_file.name, out_path)
    tmp_video_file.close()


def change_extension(file_path, new_extension):
    base = os.path.splitext(file_path)[0]
    return base + new_extension


def video_stream(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()

def cycle_video_stream(video_path, max_frames):
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        idx += 1
        if idx >= max_frames:
            break
    cap.release()


def data_to_device(data, device: Union[str, torch.device]) -> Any:
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (dict, Dict)):
        return {k: data_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, List)):
        return [data_to_device(v, device) for v in data]
    elif isinstance(data, (tuple, Tuple)):
        return tuple(data_to_device(v, device) for v in data)
    elif isinstance(data, (np.ndarray, np.generic)):
        return torch.from_numpy(data).to(device)
    else:
        return data

def data_to_numpy(data) -> Any:
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, (dict, Dict)):
        return {k: data_to_numpy(v) for k, v in data.items()}
    elif isinstance(data, (list, List)):
        return [data_to_numpy(v) for v in data]
    elif isinstance(data, (tuple, Tuple)):
        return tuple(data_to_numpy(v) for v in data)
    elif isinstance(data, (np.ndarray, np.generic)):
        return data
    else:
        return data

class LazyVideoWriter:
    def __init__(self, save_path, fps=25.0, save_images=False):
        self.writer = None
        assert save_path.endswith(".mp4"), "Only support mp4 format"
        self.save_path = save_path
        self.save_images = save_images
        self.fps = fps
        if save_images:
            self.image_dir = Path(save_path).with_suffix("")
            self.image_dir.mkdir(exist_ok=True)

    def write(self, image: np.ndarray):
        if self.writer is None:
            size = image.shape[:2][::-1]
            self.writer = cv2.VideoWriter(self.save_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, size)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if self.save_images:
            image_path = self.image_dir / f"{self.writer.get(cv2.CAP_PROP_FRAME_COUNT):06d}.png"
            cv2.imwrite(str(image_path), image_bgr)
        self.writer.write(image_bgr)

    def release(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        if exc_type is not None:
            print(f"An exception occurred: {exc_val}")
        return False


def get_total_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


def save_tensor_image(path: str, tensor: torch.Tensor):
    image = tensor.to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


async def async_video_iter(video_path, max_buffer_size=25):
    cap = cv2.VideoCapture(video_path)
    pbar = tqdm(desc="Push Pre Image")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
        await asyncio.sleep(0.04)
        pbar.update(1)
        if pbar.n >= max_buffer_size:
            break
    cap.release()
    pbar.close()


def load_audio_numpy(audio_file):
    audio, _ = librosa.load(audio_file, sr=16000, mono=True)
    audio = torch.from_numpy(audio)
    return audio


def get_device1():
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            device = "cuda:1"
        else:
            device = "cuda:0"
    else:
        device = "cpu"
    return device


def image_to_tensor(image):
    """
    [0, 255]
    """
    if isinstance(image, np.ndarray):
        image = torch.tensor(image).permute(2, 0, 1)
    assert image.dim() == 3, "Image should be 3D."
    if image.shape[2] == 3:
        image = image.permute(2, 0, 1)
    return image.float()

class AsyncTaskManager:
    logger = logging.getLogger("AsyncTaskManager")
    def __init__(self):
        self.tasks: Set[asyncio.Task] = set()
        self.loop = asyncio.get_event_loop()

    def create_task(self, coro):
        task = self.loop.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(lambda t: self.tasks.remove(t))
        self.logger.info(f"Task {task.get_name()} created.")
        return task

    def cancel_all_tasks(self):
        for task in self.tasks:
            task.cancel()
        self.logger.info("All tasks have been cancelled.")

    async def shutdown(self):
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()
        self.logger.info("Task manager has been shutdown.")


class VideoReader:
    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def read_batch(self, batch_size: int):
        frames = []
        for _ in range(batch_size):
            frame = self.read_frame()
            if frame is None:
                break
            frames.append(frame)
        return frames

    def release(self):
        self.cap.release()

    def __iter__(self):
        return self

    def __next__(self):
        frame = self.read_frame()
        if frame is None:
            self.release()
            raise StopIteration
        return frame

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()