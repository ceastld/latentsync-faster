#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
设置人脸检测模型脚本
下载并设置必要的人脸检测模型文件
"""

import os
import shutil
import urllib.request
import zipfile
import argparse

def download_file(url, target_path):
    """下载文件到指定路径"""
    print(f"下载 {url} 到 {target_path}")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    urllib.request.urlretrieve(url, target_path)

def setup_face_detection(force_download=False):
    """设置人脸检测模型"""
    # 创建目录
    face_detection_dir = os.path.join("latentsync", "face_detection")
    models_dir = os.path.join(face_detection_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # 模型文件路径
    face_detector_path = os.path.join(models_dir, "face_detector.onnx")
    landmark_detector_path = os.path.join(models_dir, "landmark_detector.onnx")
    
    # 检查是否需要从pytorch_face_landmark目录复制
    pytorch_face_landmark_dir = "pytorch_face_landmark"
    if os.path.exists(pytorch_face_landmark_dir):
        # 检查是否有face_detector.onnx
        source_face_detector = os.path.join(pytorch_face_landmark_dir, "models", "onnx", "version-RFB-320.onnx")
        if os.path.exists(source_face_detector) and (force_download or not os.path.exists(face_detector_path)):
            print(f"从 {source_face_detector} 复制到 {face_detector_path}")
            shutil.copy2(source_face_detector, face_detector_path)
        
        # 检查是否有landmark_detector.onnx
        source_landmark_detector = os.path.join(pytorch_face_landmark_dir, "onnx", "landmark_detection_56_se_external.onnx")
        if os.path.exists(source_landmark_detector) and (force_download or not os.path.exists(landmark_detector_path)):
            print(f"从 {source_landmark_detector} 复制到 {landmark_detector_path}")
            shutil.copy2(source_landmark_detector, landmark_detector_path)
    
    # 如果本地没有模型，则从GitHub下载
    if force_download or not os.path.exists(face_detector_path):
        # 下载人脸检测模型
        download_file(
            "https://github.com/cunjian/pytorch_face_landmark/raw/master/models/onnx/version-RFB-320.onnx",
            face_detector_path
        )
    
    if force_download or not os.path.exists(landmark_detector_path):
        # 下载关键点检测模型
        download_file(
            "https://github.com/cunjian/pytorch_face_landmark/raw/master/onnx/landmark_detection_56_se_external.onnx",
            landmark_detector_path
        )
    
    print("人脸检测模型设置完成")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="设置人脸检测模型")
    parser.add_argument("--force", action="store_true", help="强制重新下载模型")
    args = parser.parse_args()
    
    setup_face_detection(force_download=args.force) 