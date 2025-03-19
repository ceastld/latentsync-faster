import cv2
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from latentsync.configs.config import GLOBAL_CONFIG
from latentsync.face_detection import FaceLandmarkDetector
from latentsync.utils.affine_transform import AlignRestore, laplacianSmooth
from latentsync.utils.video import VideoReader


def draw_landmarks(image, landmarks, color=(0, 255, 0), size=3):
    """在图像上绘制关键点"""
    img = image.copy()
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(img, (int(x), int(y)), size, color, -1)
        cv2.putText(img, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return img


def test_align_warp_face(image_path=None, output_dir="outputs"):
    """测试align_warp_face函数，并可视化三个关键点"""
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 初始化检测器和变换器
    face_detector = FaceLandmarkDetector(device="cuda")
    smoother = laplacianSmooth()
    restorer = AlignRestore()
    
    # 如果提供了图像路径，则读取单张图像
    if image_path is not None:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        process_single_image(img, face_detector, smoother, restorer, output_path)
    else:
        # 否则使用视频
        reader = VideoReader(GLOBAL_CONFIG.inference.default_video_path)
        print(f"处理视频: {GLOBAL_CONFIG.inference.default_video_path}")
        print(f"总帧数: {reader.total_frames}")
        
        # 每隔10帧处理一次
        for i, frame in enumerate(reader):
            if i % 10 == 0:
                print(f"处理第 {i} 帧")
                process_single_image(frame, face_detector, smoother, restorer, output_path, frame_idx=i)
            if i >= 100:  # 只处理前100帧的部分帧
                break


def process_single_image(img, face_detector, smoother, restorer, output_path, frame_idx=None):
    """处理单张图像并可视化关键点"""
    # 检测人脸关键点
    detected_faces = face_detector.get_landmarks(img)
    if detected_faces is None:
        print("未检测到人脸")
        return
    
    lm68 = detected_faces[0]  # 第一个检测到的人脸的68个关键点
    
    # 平滑关键点
    points = smoother.smooth(lm68)
    
    # 计算三个关键点（眉毛中点和鼻子区域）
    lmk3 = np.zeros((3, 2))
    lmk3[0] = points[17:22].mean(0)  # 左眉毛中点
    lmk3[1] = points[22:27].mean(0)  # 右眉毛中点
    lmk3[2] = points[27:36].mean(0)  # 鼻子区域中点
    
    # 绘制68个关键点
    img_landmarks = draw_landmarks(img, points, color=(0, 255, 0), size=1)
    
    # 绘制3个关键对齐点（用不同颜色和大小）
    img_align_points = img_landmarks.copy()
    for i, (x, y) in enumerate(lmk3):
        cv2.circle(img_align_points, (int(x), int(y)), 5, (255, 0, 0), -1)
        cv2.putText(img_align_points, f"Point {i}", (int(x) + 10, int(y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # 执行对齐变换
    aligned_face, affine_matrix = restorer.align_warp_face(
        img.copy(), lmks3=lmk3, smooth=True, border_mode="constant"
    )
    
    # 绘制对齐模板
    template_img = np.zeros((aligned_face.shape[0], aligned_face.shape[1], 3), dtype=np.uint8) + 255
    for i, (x, y) in enumerate(restorer.face_template):
        cv2.circle(template_img, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.putText(template_img, f"Template {i}", (int(x) + 10, int(y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # 还原图像
    restored_img = restorer.restore_img(img, aligned_face, affine_matrix)
    
    # 创建4x1的网格展示图像
    plt.figure(figsize=(20, 10))
    
    # 原始图像上的68个关键点
    plt.subplot(231)
    plt.title("68个关键点")
    plt.imshow(img_landmarks)
    plt.axis('off')
    
    # 原始图像上的3个对齐关键点
    plt.subplot(232)
    plt.title("3个对齐关键点")
    plt.imshow(img_align_points)
    plt.axis('off')
    
    # 对齐后的人脸
    plt.subplot(233)
    plt.title("对齐后的人脸")
    plt.imshow(aligned_face)
    plt.axis('off')
    
    # 对齐模板
    plt.subplot(234)
    plt.title("对齐模板")
    plt.imshow(template_img)
    plt.axis('off')
    
    # 还原后的图像
    plt.subplot(235)
    plt.title("还原后的图像")
    plt.imshow(restored_img)
    plt.axis('off')
    
    # 保存图像
    suffix = f"_frame_{frame_idx}" if frame_idx is not None else ""
    plt.savefig(output_path / f"face_alignment{suffix}.png", dpi=150)
    plt.close()
    
    # 单独保存各部分图像
    cv2.imwrite(str(output_path / f"landmarks{suffix}.jpg"), cv2.cvtColor(img_landmarks, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_path / f"align_points{suffix}.jpg"), cv2.cvtColor(img_align_points, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_path / f"aligned_face{suffix}.jpg"), cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_path / f"template{suffix}.jpg"), cv2.cvtColor(template_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_path / f"restored{suffix}.jpg"), cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试人脸对齐并可视化三个关键点")
    parser.add_argument("--image", type=str, help="输入图像路径，如不提供则使用默认视频")
    parser.add_argument("--output", type=str, default="outputs", help="输出目录")
    args = parser.parse_args()
    
    test_align_warp_face(args.image, args.output) 