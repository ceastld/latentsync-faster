import cv2
import insightface
import time
import numpy as np
from tqdm import tqdm
from latentsync import *
from latentsync.utils.video import VideoReader
from latentsync.utils.timer import Timer


def test_insightface_speed():
    # 初始化 insightface，只启用需要的模块
    face_analyzer = insightface.app.FaceAnalysis(
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        allowed_modules=["detection", "landmark_3d_68", "pose"],  # 只启用需要的模块
    )
    face_analyzer.prepare(ctx_id=0, det_size=(224, 224))

    # 使用示例视频
    demo = GLOBAL_CONFIG.inference.demo_large_pose
    reader = VideoReader(demo.video_path)

    print("开始测试 insightface 检测速度...")

    @Timer("insightface_detection")
    def detect_face(frame):
        faces = face_analyzer.get(frame)
        # 只返回需要的特征
        return faces

    for i, frame in enumerate(tqdm(reader)):
        # 进行人脸检测
        faces = detect_face(frame)
        # print(faces[0])

    Timer.summary()


if __name__ == "__main__":
    Timer.enable()  # 启用计时器
    test_insightface_speed()
