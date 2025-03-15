from setuptools import setup, find_packages
import os
import re

# 从__init__.py中获取版本号
with open('latentsync/__init__.py', 'r', encoding='utf-8') as f:
    version = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read()).group(1)

# 读取requirements.txt中的依赖项
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# 读取README.md作为长描述
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="latentsync",
    version=version,
    author="LatentSync Team",
    author_email="",  # 可以添加作者邮箱
    description="Audio Conditioned Latent Diffusion Models for Lip Sync",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
        "Topic :: Multimedia :: Sound/Audio",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    # 添加入口点，方便命令行调用
    entry_points={
        "console_scripts": [
            "latentsync-infer=latentsync.inference.face_infer:main",
        ],
    },
    # 添加包数据文件
    package_data={
        "latentsync": ["configs/*.yaml", "configs/*.json"],
    },
) 