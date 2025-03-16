from accelerate.utils import set_seed as acc_seed
import numpy as np
import torch
from latentsync.inference.context import LipsyncContext
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.pipelines.lipsync_diffusion_pipeline import LipsyncDiffusionPipeline
from latentsync.whisper.whisper.audio import load_audio


def create_diffusion_pipeline(context: LipsyncContext) -> LipsyncDiffusionPipeline:
    """Create lipsync diffusion pipeline with all components"""
    # 创建模型组件
    vae = context.create_vae()
    unet = context.create_unet()
    scheduler = context.create_scheduler()
    
    # 组装模型字典用于预热
    models_dict = {
        'vae': vae,
        'unet': unet,
    }
    
    # 预热模型组件
    context.warmup_models(models_dict)
    
    # 创建并返回管道
    return LipsyncDiffusionPipeline(
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        lipsync_context=context,
    ).to(context.device)


def create_pipeline(context: LipsyncContext) -> LipsyncPipeline:
    """Create lipsync pipeline with all components"""
    # 创建各个模型组件
    vae = context.create_vae()
    audio_encoder = context.create_audio_encoder()
    unet = context.create_unet()
    scheduler = context.create_scheduler()
    
    # 创建pipeline对象
    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
        lipsync_context=context,
    ).to(context.device)
    
    # 组装模型字典用于预热 - 移除UNet预热以避免通道不匹配错误
    models_dict = {
        'vae': vae,
        'audio_encoder': audio_encoder,
        # 'unet': unet,  # 暂时移除UNet预热，避免输入格式不匹配错误
    }
    
    # 如果pipeline有face_detector属性，也添加到预热列表
    if hasattr(pipeline, 'face_processor') and hasattr(pipeline.face_processor, 'face_detector'):
        models_dict['face_detector'] = pipeline.face_processor.face_detector
    
    # 预热各组件 - UNet在实际使用时会自动获得正确的输入
    try:
        context.warmup_models(models_dict)
    except Exception as e:
        print(f"预热过程中发生错误，但将继续执行: {e}")
    
    return pipeline


def set_seed(seed: int):
    if seed != -1:
        acc_seed(seed)
    else:
        torch.seed()
        print(f"Initial seed: {torch.initial_seed()}")


def load_audio_clips(audio_path: str, samples_per_frame: int):
    audio_samples = load_audio(audio_path)
    padding_size = samples_per_frame - len(audio_samples) % samples_per_frame
    audio_samples = np.pad(
        audio_samples,
        (0, padding_size),
        mode="constant",
    )
    audio_clips = audio_samples.reshape(-1, samples_per_frame)
    return audio_clips