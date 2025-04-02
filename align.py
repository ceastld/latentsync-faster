from tqdm import tqdm
from latentsync import *
from latentsync.utils.video import VideoReader
import insightface
import numpy as np
import cv2
demo = GLOBAL_CONFIG.inference.demo_large_pose

def box_ratio(box):
    x1, y1, x2, y2 = box
    return (y2-y1) / (x2-x1)

face_analyzer = insightface.app.FaceAnalysis(
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    allowed_modules=["detection", "landmark_3d_68", "pose"],  # 只启用需要的模块
)
face_analyzer.prepare(ctx_id=0, det_size=(224, 224))

reader = VideoReader(demo.video_path)
output_path = demo.video_out_path
for i, frame in enumerate(tqdm(reader, disable=True)):
    # 使用insightface检测人脸姿态
    faces = face_analyzer.get(frame)
    if not faces:
        continue
    face = max(faces, key= lambda x: x.det_score)
    # 获取姿态信息
    pose = face.pose
    pitch, yaw, roll = pose[0], pose[1], pose[2]
    
    # 输出帧号和姿态信息
    print(f"帧 {i}: 俯仰角(pitch)={pitch:.2f}, 偏航角(yaw)={yaw:.2f}, 翻滚角(roll)={roll:.2f}")
