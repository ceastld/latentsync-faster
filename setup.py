import os
import re
import warnings
from pathlib import Path
from setuptools import setup, find_packages

# 直接在 setup.py 中指定版本号
VERSION = "0.1.2"

def setup_auxiliary_models():
    """Setup soft links for auxiliary models"""
    try:
        # Create cache directory
        cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Current working directory
        current_dir = Path.cwd()
        
        # Define the auxiliary model files to link
        auxiliary_files = [
            ("checkpoints/auxiliary/2DFAN4-cd938726ad.zip", "2DFAN4-cd938726ad.zip"),
            ("checkpoints/auxiliary/s3fd-619a316812.pth", "s3fd-619a316812.pth"),
            ("checkpoints/auxiliary/vgg16-397923af.pth", "vgg16-397923af.pth")
        ]
        
        for src_path, target_name in auxiliary_files:
            src_file = current_dir / src_path
            target_file = cache_dir / target_name
            
            # Check if source file exists
            if not src_file.exists():
                warnings.warn(f"Source file not found: {src_file}")
                continue
                
            # Check if target already exists
            if target_file.exists() or target_file.is_symlink():
                # If it's already a symlink pointing to the correct location, skip
                if target_file.is_symlink() and target_file.resolve() == src_file.resolve():
                    continue
                else:
                    warnings.warn(f"Target file already exists: {target_file}, skipping")
                    continue
            
            # Create the symlink
            try:
                target_file.symlink_to(src_file.absolute())
                print(f"Created symlink: {target_file} -> {src_file}")
            except OSError as e:
                warnings.warn(f"Failed to create symlink {target_file}: {e}")
                
    except Exception as e:
        warnings.warn(f"Error setting up auxiliary models: {e}")

# Read requirements from requirements.txt
def read_requirements():
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    try:
        with open(req_file, "r") as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            return requirements
    except Exception as e:
        raise RuntimeError("Failed to read requirements from %s: %s" % (req_file, str(e)))

# Setup auxiliary models before installation
setup_auxiliary_models()

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