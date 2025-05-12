import os
import re
from setuptools import setup, find_packages

# 直接在 setup.py 中指定版本号
VERSION = "0.1.2"

# Read requirements from requirements.txt
def read_requirements():
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    try:
        with open(req_file, "r") as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            return requirements
    except Exception as e:
        raise RuntimeError("Failed to read requirements from %s: %s" % (req_file, str(e)))

setup(
    name="latentsync",
    version=VERSION,
    description="LatentSync: A tool for lip-syncing video with audio",
    author="Pinch Research",
    author_email="info@pinchresearch.com",
    url="https://github.com/pinch-research/latentsync",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=read_requirements(),
    package_data={"latentsync": ["*.yaml","*.json"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)