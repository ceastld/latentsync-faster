from setuptools import setup, find_packages
import os

# Read version from latentsync/__init__.py
def read_version():
    version_file = os.path.join(os.path.dirname(__file__), "latentsync", "__init__.py")
    with open(version_file, "r") as f:
        exec(f.read(), globals())
    return globals()["__version__"]

setup(
    name="latentsync",
    version=read_version(),
    description="Audio Conditioned Latent Diffusion Models for Lip Sync",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Pinch",
    license="MIT",
    python_requires=">=3.8",
    packages=find_packages(include=["latentsync*"]),
    package_data={
        "latentsync": ["**/*"]
    },
    install_requires=[
        "torch>=2.2.1",
        "torchvision>=0.17.1",
        "torchaudio>=2.2.1",
        "accelerate>=1.4.0",
        "decord>=0.6.0",
        "diffusers>=0.32.2",
        "einops>=0.8.1",
        "face_alignment>=1.4.1",
        "ffmpeg_python>=0.2.0",
        "imageio>=2.37.0",
        "librosa>=0.10.2",
        "matplotlib>=3.10.1",
        "more_itertools>=10.6.0",
        "omegaconf>=2.3.0",
        "onnxruntime-gpu>=1.21.0",
        "opencv_contrib_python>=4.11.0",
        "opencv_python>=4.11.0",
        "packaging>=24.2",
        "Pillow>=11.1.0",
        "regex>=2024.11.6",
        "scipy>=1.15.2",
        "soundfile>=0.13.1",
        "tqdm>=4.67.1",
        "transformers>=4.49.0",
        "xformers>=0.0.29",
    ],
    entry_points={
        "console_scripts": [
            "latentsync-infer=latentsync.inference.face_infer:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
        "Topic :: Multimedia :: Sound/Audio",
    ],
    url="https://github.com/pinch-eng/latentsync",
    project_urls={
        "Homepage": "https://github.com/pinch-eng/latentsync",
        "Repository": "https://github.com/pinch-eng/latentsync.git",
    },
)